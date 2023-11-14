package server

// https://eli.thegreenplace.net/2019/passing-callbacks-and-pointers-to-cgo/
// https://github.com/golang/go/wiki/cgo
// https://pkg.go.dev/cmd/cgo

/*
#include <stdlib.h>
#include <stdint.h>
void * init(char * swap, char * debug);
void * initContext(
	int idx,
	char * modelName,
	int threads,
	int batch_size,
	int gpu1, int gpu2,
	int context, int predict,
	int32_t mirostat, float mirostat_tau, float mirostat_eta,
	float temperature, int topK, float topP,
	float typicalP,
	float repetition_penalty, int penalty_last_n,
	int32_t janus,
	int32_t depth,
	float scale,
	float hi,
	float lo,
	int32_t seed,
	char * debug);
int64_t doInference(
	int idx,
	void * ctx,
	char * jobID,
	char * sessionID,
	char * prompt);
void stopInference(int idx);
const char * status(char * jobID);
int64_t timing(char * jobID);
int64_t promptEval(char * jobID);
int64_t getPromptTokenCount(char * jobID);
*/
import "C"

import (
	"fmt"
	"os"
	"os/user"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	fiber "github.com/gofiber/fiber/v2"
	"github.com/goodsign/monday"
	"github.com/google/uuid"
	colorable "github.com/mattn/go-colorable"
	"github.com/mitchellh/colorstring"
	"go.uber.org/zap"

	"github.com/gotzmann/llamazoo/pkg/llama"
	"github.com/gotzmann/llamazoo/pkg/ml"
)

// TODO: Check the host:port is free before starting listening

// TODO: Helicopter View - how to work with balancers and multi-pod architectures?
// TODO: Rate Limiter based on end-user IP address
// TODO: Guard access with API Tokens
// TODO: Each use of C.CString() should be complemented with C.free() operation
// TODO: GetStatus - update partial output if processing within C++ core

// Unix timestamps VS ISO-8601 Stripe perspective:
// https://dev.to/stripe/how-stripe-designs-for-dates-and-times-in-the-api-3eoh
// TODO: UUID vs string for job ID
// TODO: Unix timestamp vs ISO for date and time

type Mode struct {
	id string
	HyperParams
}

type HyperParams struct {

	// -- Janus Sampling

	Janus uint32
	Depth uint32
	Scale float32
	Hi    float32
	Lo    float32

	Mirostat    uint32
	MirostatLR  float32 // aka eta, learning rate
	MirostatENT float32 // aka tau, target entropy
	MirostatTAU float32 // obsolete
	MirostatETA float32 // obsolete

	TopK        int
	TopP        float32
	TypicalP    float32
	Temperature float32

	RepetitionPenalty float32
	PenaltyLastN      int
}

type Model struct {
	ID     string // short internal name of the model
	Name   string // public name for humans
	Path   string // path to binary file
	Locale string

	Context unsafe.Pointer // *llama.Context

	Preamble string
	Prefix   string // prompt prefix for instruct-type models
	Suffix   string // prompt suffix

	ContextSize int
	Predict     int

	// -- Janus Sampling

	Janus uint32
	Depth uint32
	Scale float32
	Hi    float32
	Lo    float32

	Mirostat    uint32
	MirostatLR  float32 // aka eta, learning rate
	MirostatENT float32 // aka tau, target entropy
	MirostatTAU float32 // obsolete
	MirostatETA float32 // obsolete

	Temperature float32
	Temp        float32 // user-friendly naming within config
	TopK        int
	Top_K       int // user-friendly naming within config
	TopP        float32
	Top_P       float32 // user-friendly naming within config

	TypicalP float32

	RepetitionPenalty  float32
	Repetition_Penalty float32 // user-friendly naming within config
	PenaltyLastN       int
}

// TODO: Logging setup
type Config struct {
	ID    string // server key, should be unique within cluster
	Debug string // cuda, full, janus, etc

	//	Modes map[string]string // Mapping inference modes [ default, fast, ... ] to available models
	Modes []Mode

	Host string
	Port string
	Log  string // path and name of logging file

	//	AVX  bool
	//	NEON bool
	//	CUDA bool

	Swap string // path to store session files

	Pods []Pod
	//	Threads []int64 // threads count for each pod // TODO: Obsolete
	//	GPUs    [][]int // GPU split between pods // TODO: Obsolete
	// GPULayers []int   // how many layers offload to Apple GPU?

	Models []Model

	//DefaultModel string // default model ID

	Deadline int64 // deadline in seconds after which unprocessed jobs will be deleted from the queue
}

type Pod struct {
	idx int // pod index

	Threads int64  // how many threads to use
	GPUs    []int  // GPU split in percents
	Model   string // model ID within config
	//	Mode      string // TODO
	BatchSize int

	isBusy bool // do we doing some job righ not?
	isGPU  bool // pod uses GPU resources

	model *Model // real model instance
}

type Job struct {
	ID         string
	SessionID  string // ID of continuous user session in chat mode
	Status     string
	Prompt     string // exact user prompt, trimmed from spaces and newlines
	Translate  string // translation direction like "en:ru" ask translate input to EN first, then output to RU
	FullPrompt string // full prompt with prefix / suffix
	Output     string

	CreatedAt  int64
	StartedAt  int64
	FinishedAt int64

	Seed int64 // TODO: Store seed

	Mode  string
	Model string // TODO: Store model ID, "" for default should be then replaced

	PreambleTokenCount int64
	PromptTokenCount   int64 // total tokens processed (icnluding prompt)
	OutputTokenCount   int64

	PromptEval int64 // timing per token (prompt + output), ms
	TokenEval  int64 // timing per token (prompt + output), ms

	pod *Pod // we need pod.idx when stopping jobs
}

const (
	LLAMA_CPP = 0x00
	LLAMA_GO  = 0x01
	EXLLAMA   = 0x02
)

var (
	ServerMode int // LLAMA_CPP by default
	Debug      string

	Host string
	Port string

	GoShutdown bool // signal the service should go graceful shutdown

	// Data for running one model from CLI without pods instantiating
	vocab  *ml.Vocab
	model  *llama.Model
	ctx    *llama.Context
	params *llama.ModelParams

	//DefaultModel string // it's empty string "" for simple CLI mode and some unique key when working with configs

	// NB! All vars below are int64 to be used as atomic counters
	MaxThreads     int64 // used for PROD mode // TODO: detect hardware abilities automatically
	RunningThreads int64
	RunningPods    int64 // number of pods running at the moment - SHOULD BE int64 for atomic manipulations

	// FIXME TODO ASAP : Remove extra sessions from disk to prevent full disk DDoS
	Swap        string // path to store session files
	MaxSessions int    // how many sessions allowed per server, remove extra session files

	mu sync.Mutex // guards any Jobs change

	Jobs  map[string]*Job     // all seen jobs in any state
	Queue map[string]struct{} // queue of job IDs waiting for start

	Pods  []*Pod            // There N pods with some threads within as described in config
	Modes map[string]string // Each unique model might have some special [ mode ] assigned to it
	//Models      map[string][]*Model // Each unique model identified by key has N instances ready to run in pods
	Models      []*Model          // Each unique model identified by key has N instances ready to run in pods
	Sessions    map[string]string // Session store ID => HISTORY of continuous dialog with user (and the state is on the disk)
	TokensCount map[string]int    // Store tokens within each session ID => COUNT to prevent growth over context limit

	log      *zap.SugaredLogger
	deadline int64
)

func init() {
	Jobs = make(map[string]*Job, 1024)       // 1024 is like some initial space to grow
	Queue = make(map[string]struct{}, 1024)  // same here
	Sessions = make(map[string]string, 1024) // same here
	TokensCount = make(map[string]int, 1024) // same here

	// FIXME: ASAP Check those are set from within Init()
	// --- set model parameters from user settings and safe defaults
	params = &llama.ModelParams{
		Seed:       -1,
		MemoryFP16: true,
	}
}

// Init allocates contexts for independent pods
func Init(
	host, port string,
	zapLog *zap.SugaredLogger,
	pods int, threads int64,
	//gpus []int, // gpuLayers int64, // TODO: Use GPU split from config
	gpu1, gpu2 int,
	model, preamble, prefix, suffix string,
	context, predict int,
	mirostat uint32, mirostatTAU float32, mirostatETA float32,
	temperature float32, topK int, topP float32,
	typicalP float32,
	repetitionPenalty float32, penaltyLastN int,
	deadlineIn int64,
	seed uint32,
	swap string,
	debug string) {

	// ServerMode = LLAMA_CPP
	Host = host
	Port = port
	log = zapLog
	deadline = deadlineIn
	RunningPods = 0
	params.CtxSize = uint32(context)
	Pods = make([]*Pod, pods)
	Modes = map[string]string{"default": ""}
	Models = make([]*Model, 1) // TODO: N models
	Swap = swap

	Debug = debug
	//debugCUDA := 0
	//if Debug == "cuda" {
	//	debugCUDA = 1
	//}

	// --- Starting pods incorporating isolated C++ context and runtime

	for pod := 0; pod < pods; pod++ {

		MaxThreads += threads
		Pods[pod] = &Pod{
			idx: pod,

			isBusy: false,
			isGPU:  gpu1+gpu2 > 0,

			Model: model,
			// Mode:  "",

			Threads: threads,
			GPUs:    []int{gpu1, gpu2},
		}

		// Check if file exists to prevent CGO panic
		if _, err := os.Stat(model); err != nil {
			Colorize("\n[magenta][ ERROR ][white] Model not found: %s\n\n", model)
			log.Infof("[ERROR] Model not found: %s", model)
			os.Exit(0)
		}

		C.init(C.CString(swap), C.CString(Debug))

		// TODO: Refactore temp huck supporting only 2 GPUs split

		ctx := C.initContext(
			C.int(pod),
			C.CString(model),
			C.int(threads),
			C.int(0),                 // TODO: BatchSize
			C.int(gpu1), C.int(gpu2), // C.int(gpuLayers), // TODO: Support more than 2 GPUs
			C.int(context), C.int(predict),
			C.int32_t(mirostat), C.float(mirostatTAU), C.float(mirostatETA),
			C.float(temperature), C.int(topK), C.float(topP),
			C.float(typicalP),
			C.float(repetitionPenalty), C.int(penaltyLastN),
			C.int(1), C.int(200), C.float(0.936), C.float(0.982), C.float(0.948),
			C.int32_t(seed),
			C.CString(Debug),
		)

		if ctx == nil {
			Colorize("\n[magenta][ ERROR ][white] Failed to init pod #%d of total %d\n\n", pod, pods)
			os.Exit(0)
		}

		Models /*[""]*/ [pod] = &Model{
			Path:    model,
			Context: ctx,
			Locale:  "", // TODO: Set Locale

			Preamble: preamble,
			Prefix:   prefix,
			Suffix:   suffix,

			ContextSize: context,
			Predict:     predict,

			Mirostat:    mirostat,
			MirostatTAU: mirostatTAU,
			MirostatETA: mirostatETA,

			TopK:        topK,
			TopP:        topP,
			Temperature: temperature,

			RepetitionPenalty: repetitionPenalty,
			PenaltyLastN:      penaltyLastN,
		}
	}
}

// Init allocates contexts for independent pods
func InitFromConfig(conf *Config, zapLog *zap.SugaredLogger) {

	log = zapLog
	deadline = conf.Deadline

	// -- some validations TODO: move to better place

	//if conf.Pods != len(conf.Threads) {
	//	Colorize("\n[magenta][ ERROR ][white] Please fix config! Treads array should have numbers for each pod of total %d\n\n", conf.Pods)
	//	os.Exit(0)
	//}

	//for conf.Pods != len(conf.GPUs) {
	//	Colorize("\n[magenta][ ERROR ][white] Please fix config! Set GPU split for each pod\n\n")
	//	os.Exit(0)
	//}

	//defaultModelSet := false
	//for mode, model := range conf.Modes {
	//	if mode == "default" {
	//		defaultModelSet = true
	//		DefaultModel = model
	//	}
	//}

	//if !defaultModelSet {
	//	Colorize("\n[magenta][ ERROR ][white] Default model is not set with config [ modes ] section!\n\n")
	//	log.Infof("[ERROR] Default model is not set with config [ modes ] section!")
	//	os.Exit(0)
	//}

	// -- init golbal settings

	ServerMode = LLAMA_CPP
	Host = conf.Host
	Port = conf.Port
	Pods = make([]*Pod, len(conf.Pods))
	Models = make([]*Model, len(conf.Models))
	Swap = conf.Swap

	Debug = conf.Debug
	//debugCUDA := 0
	//if Debug == "cuda" {
	//	debugCUDA = 1
	//}

	// -- Init all pods and models to run inside each pod - so having N * M total models ready to work

	//for pod, threads := range conf.Threads {
	for idx, pod := range conf.Pods {

		MaxThreads += pod.Threads

		isGPU := false
		for _, layers := range pod.GPUs {
			if layers > 0 {
				isGPU = true
			}
		}

		Pods[idx] = &Pod{
			idx: idx,

			isBusy: false,
			isGPU:  isGPU,

			Threads:   pod.Threads,
			BatchSize: pod.BatchSize,
			GPUs:      pod.GPUs,

			Model: pod.Model,
			// Mode:  pod.Mode,
		}

		for _, model := range conf.Models {

			// --- Allow user home dir resolve with tilde ~
			// TODO: // Use strings.HasPrefix so we don't match paths like "/something/~/something/"

			path := model.Path
			if strings.HasPrefix(path, "~/") {
				usr, _ := user.Current()
				dir := usr.HomeDir
				path = filepath.Join(dir, path[2:])
			}

			// Check if file exists to prevent CGO panic
			if _, err := os.Stat(path); err != nil {
				Colorize("\n[magenta][ ERROR ][white] Model not found: %s\n\n", path)
				log.Infof("[ERROR] Model not found: %s", path)
				os.Exit(0)
			}

			C.init(C.CString(Swap), C.CString(Debug))

			// TODO: Refactore temp huck supporting only 2 GPUs split

			gpu1 := 0
			gpu2 := 0

			tau := model.MirostatTAU
			if model.MirostatENT != 0 {
				tau = model.MirostatENT
			}
			eta := model.MirostatETA
			if model.MirostatLR != 0 {
				eta = model.MirostatLR
			}

			if len(pod.GPUs) > 0 {
				gpu1 = pod.GPUs[0]
				if len(pod.GPUs) > 1 {
					gpu2 = pod.GPUs[1]
				}
			}

			ctx := C.initContext(
				C.int(idx),
				C.CString(path),
				C.int(pod.Threads),
				C.int(pod.BatchSize),
				// C.int(conf.GPUs[pod]), C.int(conf.GPULayers[pod]),
				C.int(gpu1), C.int(gpu2),
				C.int(model.ContextSize), C.int(model.Predict),
				C.int32_t(model.Mirostat), C.float(tau), C.float(eta),
				C.float(model.Temperature), C.int(model.TopK), C.float(model.TopP),
				C.float(model.TypicalP),
				C.float(model.RepetitionPenalty), C.int(model.PenaltyLastN),
				C.int(model.Janus), C.int(model.Depth), C.float(model.Scale), C.float(model.Hi), C.float(model.Lo),
				C.int32_t(-1),
				C.CString(Debug),
			)

			if ctx == nil {
				Colorize("\n[magenta][ ERROR ][white] Failed to init pod for model [ %s ]\n\n", model.ID)
				os.Exit(0)
			}

			Models[idx] = &Model{
				ID:     model.ID,
				Name:   model.Name,
				Locale: model.Locale,

				Path:    model.Path,
				Context: ctx,

				Preamble: model.Preamble,
				Prefix:   model.Prefix,
				Suffix:   model.Suffix,

				ContextSize: model.ContextSize,
				Predict:     model.Predict,

				// FIXME
				Mirostat:    model.Mirostat,
				MirostatTAU: tau,
				MirostatETA: eta,

				Janus: model.Janus,
				Depth: model.Depth,
				Scale: model.Scale,
				Hi:    model.Hi,
				Lo:    model.Lo,

				TopK:        model.TopK,
				TopP:        model.TopP,
				Temperature: model.Temperature,

				RepetitionPenalty: model.RepetitionPenalty,
				PenaltyLastN:      model.PenaltyLastN,
			}
		}
	}
}

// --- init and run Fiber server

func Run(showStatus bool) {

	app := fiber.New(fiber.Config{
		DisableStartupMessage: true,
	})

	app.Post("/jobs/", NewJob)
	app.Delete("/jobs/:id", StopJob)
	app.Get("/jobs/status/:id", GetJobStatus)
	app.Get("/jobs/:id", GetJob)

	app.Get("/health", GetHealth)

	go Engine(app)

	if showStatus {
		Colorize("\n[light_magenta][ INIT ][light_blue] REST API running on [light_magenta]%s:%s", Host, Port)
	}

	log.Infof("[START] REST API running on %s:%s", Host, Port)

	err := app.Listen(Host + ":" + Port)
	if err != nil {
		Colorize("\n[light_magenta][ERROR][light_blue] Can't start REST API on [light_magenta]%s:%s", Host, Port)
		log.Infof("[ERROR] Can't start REST API on %s:%s", Host, Port)
	}
}

// --- evergreen Engine looking for job queue and starting up to MaxPods workers

func Engine(app *fiber.App) {

	for {

		for jobID := range Queue {

			// TODO: MaxThreads instead of MaxPods
			// FIXME: Move to outer loop?

			// simple mode with settings from CLI
			//if MaxPods > 0 && RunningPods >= MaxPods {
			//	continue
			//}

			// production mode with settings from config file
			// TODO: >= MaxThreads + pod.Model.Threads
			if RunningThreads >= MaxThreads {
				continue
			}

			// TODO: Better to store model name right there with JobID to avoid locking
			/////mu.Lock()
			/////model := Jobs[jobID].Model
			/////mu.Unlock()

			/////if MaxThreads > 0 && len(IdlePods[model]) == 0 {
			/////	continue
			/////}

			// -- move job from waiting queue to processing and assign it pod from idle pool
			// TODO: Use different mutexes for Jobs map, Pods map and maybe for atomic counters

			now := time.Now().UnixMilli()

			mu.Lock() // -- locked

			delete(Queue, jobID)

			// ignore jobs placed more than 3 minutes ago
			if deadline > 0 && (now-Jobs[jobID].CreatedAt) > deadline*1000 {
				delete(Jobs, jobID)
				mu.Unlock()
				log.Infow("[JOB] Job was deleted after deadline", zap.String("jobID", jobID))
				continue
			}

			Jobs[jobID].Status = "processing"

			var pod *Pod
			for idx := range Pods {
				pod = Pods[idx]
				if pod.isBusy {
					continue
				}
				pod.isBusy = true
				// "load" the model into pod
				pod.model = Models[idx]
				break
			}

			mu.Unlock() // -- unlocked

			if pod == nil {
				// FIXME: Something really wrong going here! We need to fix this ASAP
				// TODO: Log this case!
				Colorize("\n[magenta][ ERROR ][white] Failed to get idle pod!\n\n")
				continue
			}

			// FIXME: Check RunningPods one more time?
			// TODO: Is it make sense to use atomic over just mutex here?
			atomic.AddInt64(&RunningPods, 1)
			atomic.AddInt64(&RunningThreads, pod.Threads)

			go Do(jobID, pod)
		}

		if GoShutdown && len(Queue) == 0 && RunningThreads == 0 {
			app.Shutdown()
			break
		}

		// TODO: Sync over channels
		time.Sleep(50 * time.Millisecond)
	}
}

// --- worker doing the "job" of transforming boring prompt into magic output

func Do(jobID string, pod *Pod) {

	// TODO: still need mutext for subtract both counters at the SAME time
	defer atomic.AddInt64(&RunningPods, -1)
	defer atomic.AddInt64(&RunningThreads, -pod.Threads)
	defer runtime.GC() // TODO: GC or not GC?

	// TODO: Proper logging
	// fmt.Printf("\n[ PROCESSING ] Starting job # %s", jobID)
	now := time.Now().UnixMilli()
	//isGPU := pod.isGPU

	mu.Lock() // --

	sessionID := Jobs[jobID].SessionID
	Jobs[jobID].pod = pod
	Jobs[jobID].Model = pod.model.ID
	Jobs[jobID].StartedAt = now
	//Jobs[jobID].Timings = make([]int64, 0, 1024) // Reserve reasonable space (like context size) for storing token evaluation timings
	// TODO: Play with prompt without leading space
	//prompt := " " + Jobs[jobID].Prompt // add a space to match LLaMA tokenizer behavior
	// TODO: Allow setting prefix/suffix from CLI
	// TODO: Implement translation for prompt elsewhere

	// -- check if we are possibly going to grow out of context limit [ 2048 tokens ] and need to drop session data

	if sessionID != "" {

		var SessionFile string

		if !pod.isGPU && Swap != "" && sessionID != "" {
			SessionFile = Swap + "/" + sessionID
		}

		// -- null the session when near the context limit (allow up to 1/2 of max predict size)
		// TODO: We need a better (smart) context data handling here

		if (TokensCount[sessionID] + (pod.model.Predict / 2) + 4) > pod.model.ContextSize {

			Sessions[sessionID] = ""
			TokensCount[sessionID] = 0

			if !pod.isGPU && SessionFile != "" {
				os.Remove(SessionFile)
			}
		}
	}

	// -- Inject context vars: ${DATE}, etc
	locale := monday.LocaleEnUS
	if pod.model.Locale != "" {
		locale = pod.model.Locale
	}
	date := monday.Format(time.Now(), "Monday 2 January 2006", monday.Locale(locale))
	date = strings.ToLower(date)
	preamble := strings.Replace(pod.model.Preamble, "${DATE}", date, 1)
	// fmt.Printf("\nPREAMBLE: %s", preamble) // DEBUG
	// --

	prompt := Jobs[jobID].Prompt
	fullPrompt := pod.model.Prefix + prompt + pod.model.Suffix
	history := Sessions[sessionID] // empty for 1) the first iteration, 2) after the limit was reached and 3) when sessions do not stored at all

	if history == "" {
		fullPrompt = preamble + fullPrompt
	} else {
		fullPrompt = history + fullPrompt
	}
	//fullPrompt = strings.Replace(fullPrompt, `\n`, "\n", -1)

	Jobs[jobID].FullPrompt = fullPrompt

	mu.Unlock() // --

	// if ServerMode == LLAMA_CPP { // --- use llama.cpp backend

	// FIXME: Do not work as expected. Empty file rise CGO exception here
	//        error loading session file: unexpectedly reached end of file
	//        do_inference: error: failed to load session file './sessions/5fb8ebd0-e0c9-4759-8f7d-35590f6c9f01'

	/*

		if _, err := os.Stat(SessionFile); err != nil {
			if os.IsNotExist(err) {
				_, err = os.Create(SessionFile)
				if err != nil {
					Colorize("\n[magenta][ ERROR ][white] Can't create session file: %s\n\n", SessionFile)
					log.Infof("[ERROR] Can't create session file: %s", SessionFile)
					os.Exit(0)
				}
			} else {
				Colorize("\n[magenta][ ERROR ][white] Some problems with session file: %s\n\n", SessionFile)
				log.Infof("[ERROR] Some problems with session file: %s", SessionFile)
				os.Exit(0)
			}
		}

	*/

	// FIXME: if model hparams were changes, session files are obsolete

	// do_inference: attempting to load saved session from './session.data.bin'
	// llama_load_session_file_internal : model hparams didn't match from session file!
	// do_inference: error: failed to load session file './session.data.bin'

	outputTokenCount := C.doInference(C.int(pod.idx), pod.model.Context, C.CString(jobID), C.CString(sessionID), C.CString(fullPrompt))
	result := C.GoString(C.status(C.CString(jobID)))
	promptTokenCount := C.getPromptTokenCount(C.CString(jobID))

	//Colorize("\n=== HISTORY ===\n%s\n", history)
	//Colorize("\n=== FULL PROMPT ===\n%s\n", fullPrompt)
	//Colorize("\n=== RESULT ===\n%s\n", result)

	// LLaMA(cpp) tokenizer might add leading space
	if len(result) > 0 && len(fullPrompt) > 0 && fullPrompt[0] != ' ' && result[0] == ' ' {
		result = result[1:]
	}

	// Save exact result as history for future session work if storage enabled
	if sessionID != "" {
		mu.Lock() // --
		Sessions[sessionID] = result
		TokensCount[sessionID] += int(outputTokenCount)
		mu.Unlock() // --
	}

	//if strings.HasPrefix(result, fullPrompt) {
	//	result = result[len(fullPrompt):]
	//}
	//result = strings.Trim(result, "\n ")
	//Colorize("\n=== RESULT AFTER ===\n%s\n", result)

	// NB! Do show nothing if output is shorter than the whole history before
	if len(result) <= len(fullPrompt) {
		//mt.Printf("\n===> ZEROING")
		result = ""
	} else {
		result = result[len(fullPrompt):]
		result = strings.Trim(result, "\n ")
	}

	//fmt.Printf("\n\nRESULT: [[[%s]]]", result)
	//fmt.Printf("\n\nPROMPT: [[[%s]]]", fullPrompt)

	now = time.Now().UnixMilli()
	promptEval := int64(C.promptEval(C.CString(jobID)))
	eval := int64(C.timing(C.CString(jobID)))

	mu.Lock() // --

	Jobs[jobID].FinishedAt = now
	if Jobs[jobID].Status != "stopped" {
		Jobs[jobID].Status = "finished"
	}

	// FIXME ASAP : Log all meaninful details !!!
	Jobs[jobID].PromptTokenCount = int64(promptTokenCount)
	Jobs[jobID].OutputTokenCount = int64(outputTokenCount)
	Jobs[jobID].PromptEval = promptEval
	Jobs[jobID].TokenEval = eval
	Jobs[jobID].Output = result
	Jobs[jobID].pod = nil
	pod.isBusy = false

	mu.Unlock() // --

	// NB! Avoid division by zero
	var inTPS, outTPS int64
	if promptEval != 0 {
		inTPS = 1000 / promptEval
	}
	if eval != 0 {
		outTPS = 1000 / eval
	}

	log.Infow(
		"[JOB] Job was finished",
		"jobID", jobID,
		"inLen", promptTokenCount,
		"outLen", outputTokenCount,
		"inMS", promptEval,
		"outMS", eval,
		"inTPS", inTPS,
		"outTPS", outTPS,
		"prompt", prompt,
		"output", result,
		// "fullPrompt", fullPrompt,
	)

	/*
	   } else { // --- use llama.go framework

	   	// tokenize the prompt
	   	embdPrompt := ml.Tokenize(vocab, fullPrompt, true)

	   	// ring buffer for last N tokens
	   	lastNTokens := ring.New(int(params.CtxSize))

	   	// method to append a token to the ring buffer
	   	appendToken := func(token uint32) {
	   		lastNTokens.Value = token
	   		lastNTokens = lastNTokens.Next()
	   	}

	   	// zeroing the ring buffer
	   	for i := 0; i < int(params.CtxSize); i++ {
	   		appendToken(0)
	   	}

	   	evalCounter := 0
	   	tokenCounter := 0
	   	pastCount := uint32(0)
	   	consumedCount := uint32(0)           // number of tokens, already processed from the user prompt
	   	remainedCount := params.PredictCount // how many tokens we still need to generate to achieve predictCount
	   	embd := make([]uint32, 0, params.BatchSize)

	   	evalPerformance := make([]int64, 0, params.PredictCount)
	   	samplePerformance := make([]int64, 0, params.PredictCount)
	   	fullPerformance := make([]int64, 0, params.PredictCount)

	   	// new context opens sync channel and starts workers for tensor compute
	   	ctx := llama.NewContext(model, params)

	   	for remainedCount > 0 {

	   		// TODO: Store total time of evaluation and average per token + token count
	   		start := time.Now().UnixNano()

	   		if len(embd) > 0 {

	   			// infinite text generation via context swapping
	   			// if we run out of context:
	   			// - take the n_keep first tokens from the original prompt (via n_past)
	   			// - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch

	   			if pastCount+uint32(len(embd)) > params.CtxSize {
	   				leftCount := pastCount - params.KeepCount
	   				pastCount = params.KeepCount

	   				// insert n_left/2 tokens at the start of embd from last_tokens
	   				// embd = append(lastNTokens[:leftCount/2], embd...)
	   				embd = append(llama.ExtractTokens(lastNTokens.Move(-int(leftCount/2)), int(leftCount/2)), embd...)
	   			}

	   			evalStart := time.Now().UnixNano()
	   			if err := llama.Eval(ctx, vocab, model, embd, pastCount, params); err != nil {
	   				// TODO: Finish job properly with [failed] status
	   			}
	   			evalPerformance = append(evalPerformance, time.Now().UnixNano()-evalStart)
	   			evalCounter++
	   		}

	   		pastCount += uint32(len(embd))
	   		embd = embd[:0]

	   		if int(consumedCount) < len(embdPrompt) {

	   			for len(embdPrompt) > int(consumedCount) && len(embd) < int(params.BatchSize) {

	   				embd = append(embd, embdPrompt[consumedCount])
	   				appendToken(embdPrompt[consumedCount])
	   				consumedCount++
	   			}

	   		} else {

	   			//if params.IgnoreEOS {
	   			//	Ctx.Logits[ml.TOKEN_EOS] = 0
	   			//}

	   			sampleStart := time.Now().UnixNano()
	   			id := llama.SampleTopPTopK( / * ctx, * / ctx.Logits,
	   				lastNTokens, params.PenaltyLastN,
	   				params.TopK, params.TopP,
	   				params.Temp, params.RepeatPenalty)
	   			samplePerformance = append(samplePerformance, time.Now().UnixNano()-sampleStart)

	   			appendToken(id)

	   			// replace end of text token with newline token when in interactive mode
	   			//if id == ml.TOKEN_EOS && params.Interactive && !params.Instruct {
	   			//	id = ml.NewLineToken
	   			//}

	   			embd = append(embd, id) // add to the context

	   			remainedCount-- // decrement remaining sampling budget
	   		}

	   		fullPerformance = append(fullPerformance, time.Now().UnixNano()-start)

	   		// skip adding the whole prompt to the output if processed at once
	   		if evalCounter == 0 && int(consumedCount) == len(embdPrompt) {
	   			continue
	   		}

	   		// --- assemble the final ouptut, EXCLUDING the prompt

	   		for _, id := range embd {

	   			tokenCounter++
	   			token := ml.Token2Str(vocab, id) // TODO: Simplify

	   			mu.Lock()
	   			Jobs[jobID].Output += token
	   			mu.Unlock()
	   		}
	   	}

	   	// close sync channel and stop compute workers
	   	ctx.ReleaseContext()

	   	mu.Lock()
	   	Jobs[jobID].FinishedAt = time.Now().UnixMilli()
	   	// FIXME: Clean output from prefix/suffix for instruct models!
	   	Jobs[jobID].Output = strings.Trim(Jobs[jobID].Output, "\n ")
	   	Jobs[jobID].Status = "finished"
	   	mu.Unlock()

	   	//if ml.DEBUG {
	   	Colorize("\n\n=== EVAL TIME | ms ===\n\n")
	   	for _, time := range evalPerformance {
	   		Colorize("%d | ", time/1_000_000)
	   	}

	   	Colorize("\n\n=== SAMPLING TIME | ms ===\n\n")
	   	for _, time := range samplePerformance {
	   		Colorize("%d | ", time/1_000_000)
	   	}

	   	Colorize("\n\n=== FULL TIME | ms ===\n\n")
	   	for _, time := range fullPerformance {
	   		Colorize("%d | ", time/1_000_000)
	   	}

	   	avgEval := int64(0)
	   	for _, time := range fullPerformance {
	   		avgEval += time / 1_000_000
	   	}
	   	avgEval /= int64(len(fullPerformance))

	   	Colorize(
	   		"\n\n[light_magenta][ HALT ][white] Time per token: [light_cyan]%d[white] ms | Tokens per second: [light_cyan]%.2f\n\n",
	   		avgEval,
	   		float64(1000)/float64(avgEval))
	   	//}

	   	// TODO: Proper logging
	   	// fmt.Printf("\n[ PROCESSING ] Finishing job # %s", jobID)

	   }
	*/
}

// --- Place new job into queue

func PlaceJob(jobID, mode, model, sessionID, prompt, translate string) {

	timing := time.Now().UnixMilli()

	mu.Lock()

	Jobs[jobID] = &Job{
		ID:        jobID,
		Mode:      mode,
		Model:     model,
		SessionID: sessionID,
		Prompt:    prompt,
		Translate: translate,
		Status:    "queued",
		CreatedAt: timing,
	}

	Queue[jobID] = struct{}{}

	mu.Unlock()
}

// --- POST /jobs
//
//	{
//	    "id": "5fb8ebd0-e0c9-4759-8f7d-35590f6c9fcb",
//      "model": "airoboros-7b",
//	    "prompt": "Why Golang is so popular?"
//	}

func NewJob(ctx *fiber.Ctx) error {

	if GoShutdown {
		return ctx.
			Status(fiber.StatusServiceUnavailable).
			SendString("Service shutting down...")
	}

	payload := struct {
		ID        string `json:"id"`
		Session   string `json:"session,omitempty"`
		Mode      string `json:"mode,omitempty"`
		Model     string `json:"model,omitempty"`
		Prompt    string `json:"prompt"`
		Translate string `json:"translate"`
	}{}

	if err := ctx.BodyParser(&payload); err != nil {
		// TODO: Proper error handling
	}

	// -- normalize prompt

	payload.Prompt = strings.Trim(payload.Prompt, "\n ")
	//payload.Mode = strings.Trim(payload.Mode, "\n ")
	//payload.Model = strings.Trim(payload.Model, "\n ")
	//payload.Translate = strings.Trim(payload.Translate, "\n ")

	// -- validate prompt

	if payload.Mode != "" {
		if _, ok := Modes[payload.Mode]; !ok {
			return ctx.
				Status(fiber.StatusBadRequest).
				SendString("Wrong mode!")
		}
	}

	//if payload.Model != "" {
	//	if _, ok := Models[payload.Model]; !ok {
	//		return ctx.
	//			Status(fiber.StatusBadRequest).
	//			SendString("Wrong model name!")
	//	}
	//}

	if _, err := uuid.Parse(payload.ID); err != nil {
		return ctx.
			Status(fiber.StatusBadRequest).
			SendString("Wrong requerst id, please use UUIDv4 format!")
	}

	mu.Lock()
	if _, ok := Jobs[payload.ID]; ok {
		mu.Unlock()
		return ctx.
			Status(fiber.StatusBadRequest).
			SendString("Request with the same id is already processing!")
	}
	mu.Unlock()

	// FIXME: Return check!
	// TODO: Tokenize and check for max tokens properly
	//	if len(payload.Prompt) >= int(params.CtxSize)*3 {
	//		return ctx.
	//			Status(fiber.StatusBadRequest).
	//			SendString(fmt.Sprintf("Prompt length is more than allowed %d tokens!", params.CtxSize))
	//	}

	//if payload.Model != "" {
	// FIXME: Refactor ASAP
	/////if _, ok := Pods[payload.Model]; !ok {
	/////	return ctx.
	/////		Status(fiber.StatusBadRequest).
	/////		SendString(fmt.Sprintf("Model with name '%s' is not found!", payload.Model))
	/////}
	//} else {
	//	payload.Model = DefaultModel
	//}

	// FIXME ASAP : Use payload Model and Mode selectors !!!
	payload.Model = "" // TODO: DefaultModel

	PlaceJob(payload.ID, payload.Mode, payload.Model, payload.Session, payload.Prompt, payload.Translate)

	log.Infow("[JOB] New job just queued", "jobID", payload.ID, "mode", payload.Mode, "model", payload.Model, "session", payload.Session, "prompt", payload.Prompt)

	// TODO: Guard with mutex Jobs[payload.ID] access
	// TODO: Return [model] and [session] if not empty
	return ctx.JSON(fiber.Map{
		"id": payload.ID,
		//"session": payload.Session,
		//"model":   payload.Model,
		//"prompt": payload.Prompt,
		//"created": Jobs[payload.ID].CreatedAt,
		//"started":  Jobs[payload.ID].StartedAt,
		//"finished": Jobs[payload.ID].FinishedAt,
		//"model":    "model-xx", // TODO: Real model ID
		//"source":   "api",      // TODO: Enum for sources
		//"status": Jobs[payload.ID].Status,
		"status": "queued",
	})
}

// --- DELETE /jobs/:id

func StopJob(ctx *fiber.Ctx) error {

	jobID := ctx.Params("id")

	if _, err := uuid.Parse(jobID); err != nil {
		return ctx.
			Status(fiber.StatusBadRequest).
			SendString("Wrong UUID4 id for request!")
	}

	mu.Lock() // --

	if _, ok := Jobs[jobID]; !ok {
		mu.Unlock()
		return ctx.
			Status(fiber.StatusBadRequest).
			SendString("Request ID was not found!")
	}

	if Jobs[jobID].Status == "queued" {
		delete(Queue, jobID)
	}

	Jobs[jobID].Status = "stopped"

	if Jobs[jobID].pod != nil {
		C.stopInference(C.int(Jobs[jobID].pod.idx))
	}

	mu.Unlock() // --

	log.Infow("[JOB] Job was stopped", "jobID", jobID)

	return ctx.JSON(fiber.Map{
		"status": "stopped",
	})
}

// --- GET /jobs/status/:id

func GetJobStatus(ctx *fiber.Ctx) error {

	id := ctx.Params("id")

	if _, err := uuid.Parse(id); err != nil {
		return ctx.
			Status(fiber.StatusBadRequest).
			SendString("Wrong ID format in request!")
	}

	// TODO: Guard with mutex
	if _, ok := Jobs[id]; !ok {
		return ctx.
			Status(fiber.StatusBadRequest).
			SendString("Requested ID was not found!")
	}

	// TODO: Guard with mutex
	return ctx.JSON(fiber.Map{
		"status": Jobs[id].Status,
	})
}

// --- GET /jobs/:id

func GetJob(ctx *fiber.Ctx) error {

	jobID := ctx.Params("id")

	if _, err := uuid.Parse(jobID); err != nil {
		return ctx.
			Status(fiber.StatusBadRequest).
			SendString("Wrong ID format in request!")
	}

	if _, ok := Jobs[jobID]; !ok {
		return ctx.
			Status(fiber.StatusNotFound).
			SendString("Requested ID was not found!")
	}

	mu.Lock() // --
	status := Jobs[jobID].Status
	prompt := Jobs[jobID].Prompt
	fullPrompt := Jobs[jobID].FullPrompt // we need the full prompt with prefix and suffix here
	output := Jobs[jobID].Output
	//created := Jobs[jobID].CreatedAt
	//finished := Jobs[jobID].FinishedAt
	//model := Jobs[jobID].Model
	mu.Unlock() // --

	//fullPrompt = strings.Trim(fullPrompt, "\n ")

	if status == "processing" {
		output = C.GoString(C.status(C.CString(jobID)))

		// LLaMA(cpp) tokenizer might add leading space
		if len(output) > 0 && len(fullPrompt) > 0 && fullPrompt[0] != ' ' && output[0] == ' ' {
			output = output[1:]
		}

		//fmt.Printf("\n\nOUTPUT: [[[%s]]]", output)
		//fmt.Printf("\n\nPROMPT: [[[%s]]]", fullPrompt)

		// NB! Do show nothing if output is shorter than the whole history before
		if len(output) <= len(fullPrompt) {
			output = ""
		} else {
			output = output[len(fullPrompt):]
			output = strings.Trim(output, "\n ")
		}

		//fmt.Printf("\n\nOUTPUT: [[[%s]]]", output)
		//fmt.Printf("\n\nPROMPT: [[[%s]]]", fullPrompt)
	}

	return ctx.JSON(fiber.Map{
		"id":     jobID,
		"status": status,
		"prompt": prompt,
		"output": output,
		//"created": created,
		//"started":  Jobs[jobID].StartedAt,
		//"finished": finished,
		//"model":    model,
	})
}

// --- GET /health

func GetHealth(ctx *fiber.Ctx) error {

	cpuPercent := float32(RunningThreads) / float32(MaxThreads)

	return ctx.JSON(fiber.Map{
		"podCount": len(Pods),
		// fmt.Sprintf("%.2f", float32(RunningThreads)/float32(MaxThreads)*100)
		"cpuLoad": cpuPercent,
		"gpuLoad": 0.0, // TODO ASAP
	})
}

// Colorize is a wrapper for colorstring.Color() and fmt.Fprintf()
// Join colorstring and go-colorable to allow colors both on Mac and Windows
// TODO: Implement as a small library
func Colorize(format string, opts ...interface{}) (n int, err error) {
	//if !doPrint {
	//	return
	//}
	var DefaultOutput = colorable.NewColorableStdout()
	return fmt.Fprintf(DefaultOutput, colorstring.Color(format), opts...)
}
