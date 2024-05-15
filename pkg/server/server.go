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
	int gpu1, int gpu2, int gpu3, int gpu4,
	int context, int predict,
	int32_t mirostat, float mirostat_tau, float mirostat_eta,
	float temperature, int topK, float topP,
	float typicalP,
	float repetition_penalty, int penalty_last_n,
	int32_t janus, int32_t depth, float scale, float hi, float lo,
	uint32_t seed,
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
	"strconv"
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

	"github.com/gotzmann/booster/pkg/llama"
	"github.com/gotzmann/booster/pkg/ml"
)

const LLAMA_DEFAULT_SEED = uint32(0xFFFFFFFF)

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
	//HyperParams
}

type Model struct {
	ID string // short internal name of the model

	Name string // public name for humans
	Path string // path to binary file
	// Locale string

	Context unsafe.Pointer // *llama.Context

	ContextSize int
	Predict     int
}

type Prompt struct {
	ID string

	// -- older format

	Preamble string
	Prefix   string // prompt prefix for instruct-type models
	Suffix   string // prompt suffix

	// -- new format

	Locale    string
	System    string
	User      string
	Assistant string
}

type Sampling struct {
	ID string

	// -- Janus

	Janus uint32
	Depth uint32
	Scale float32
	Hi    float32
	Lo    float32

	// -- Mirostat

	Mirostat    uint32
	MirostatLR  float32 // aka eta, learning rate
	MirostatENT float32 // aka tau, target entropy
	///// MirostatTAU float32 // obsolete
	///// MirostatETA float32 // obsolete

	// -- Basic

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

	Pods      map[string]*Pod
	Models    map[string]*Model
	Prompts   map[string]*Prompt
	Samplings map[string]*Sampling

	//DefaultModel string // default model ID

	Deadline int64 // deadline in seconds after which unprocessed jobs will be deleted from the queue
}

type Pod struct {
	ID  string // pod name
	idx int    // pod index

	Threads  int64  // how many threads to use
	GPUs     []int  // GPU split in percents
	Model    string // model ID within config
	Prompt   string // TODO: Allow any prompt on request
	Sampling string // sampling ID within config (TODO: Allow any sampling method on request)
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

	Pod *Pod // we need pod.idx when stopping jobs
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

	Pods      map[string]*Pod   // There N pods with some threads within as described in config
	Models    map[string]*Model // Each unique model identified by key has N instances ready to run in pods
	Prompts   map[string]*Prompt
	Samplings map[string]*Sampling

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
	gpu1, gpu2, gpu3, gpu4 int, //gpus []int, // gpuLayers int64, // TODO: Use GPU split from config
	model, preamble, prefix, suffix string,
	context, predict int,
	mirostat uint32, mirostatENT float32, mirostatLR float32,
	temperature float32, topK int, topP float32,
	typicalP float32,
	repetitionPenalty float32, penaltyLastN int,
	deadlineIn int64,
	seed uint32,
	swap string,
	debug string) {

	Host = host
	Port = port
	log = zapLog
	deadline = deadlineIn
	RunningPods = 0
	params.CtxSize = uint32(context)

	Pods = make(map[string]*Pod, pods)
	Models = make(map[string]*Model, 1)       // TODO: N
	Prompts = make(map[string]*Prompt, 1)     // TODO: N
	Samplings = make(map[string]*Sampling, 1) // TODO: N

	Prompts[""] = &Prompt{} // FIXME

	Swap = swap
	Debug = debug

	// --- Starting pods incorporating isolated C++ context and runtime

	for podNum := 0; podNum < pods; podNum++ {

		pod := strconv.Itoa(podNum)
		MaxThreads += threads

		Pods[pod] = &Pod{
			idx: podNum,

			isBusy: false,
			isGPU:  gpu1+gpu2+gpu3+gpu4 > 0,

			Model:    model,
			Prompt:   "", // TODO FIXME
			Sampling: "", // TODO FIXME

			Threads: threads,
			GPUs:    []int{gpu1, gpu2, gpu3, gpu4},
		}

		// Check if file exists to prevent CGO panic
		if _, err := os.Stat(model); err != nil {
			Colorize("\n[magenta][ ERROR ][white] Model not found: %s\n\n", model)
			log.Infof("[ ERROR ] Model not found: %s", model)
			os.Exit(0)
		}

		C.init(C.CString(swap), C.CString(Debug))

		ctx := C.initContext(
			C.int(podNum),
			C.CString(model),
			C.int(threads),
			C.int(0),                                           // TODO: BatchSize
			C.int(gpu1), C.int(gpu2), C.int(gpu3), C.int(gpu4), // C.int(gpuLayers), // FIXME ASAP: TODO: Support more than 4 GPUs
			C.int(context), C.int(predict),
			C.int32_t(mirostat), C.float(mirostatENT), C.float(mirostatLR),
			C.float(temperature), C.int(topK), C.float(topP),
			C.float(typicalP),
			C.float(repetitionPenalty), C.int(penaltyLastN),
			C.int(1) /* Janus Version */, C.int(200) /* depth */, C.float(0.936) /* scale */, C.float(0.982) /* hi */, C.float(0.948), /* lo */
			C.uint32_t(seed),
			C.CString(Debug),
		)

		if ctx == nil {
			Colorize("\n[magenta][ ERROR ][white] Failed to init pod #%d of total %d\n\n", pod, pods)
			os.Exit(0)
		}

		Models[pod] = &Model{
			Path:        model,
			Context:     ctx,
			ContextSize: context,
			Predict:     predict,
		}

		Prompts[pod] = &Prompt{
			Locale: "", // TODO: Set Locale

			Preamble: preamble,
			Prefix:   prefix,
			Suffix:   suffix,

			System:    "", // TODO: prompt.System,
			User:      "", // TODO: prompt.User,
			Assistant: "", // TODO: prompt.Assistant,
		}

		Samplings[pod] = &Sampling{

			Mirostat:    mirostat,
			MirostatENT: mirostatENT,
			MirostatLR:  mirostatLR,

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
	//	log.Infof("[ ERROR ] Default model is not set with config [ modes ] section!")
	//	os.Exit(0)
	//}

	// -- init golbal settings

	ServerMode = LLAMA_CPP
	Host = conf.Host
	Port = conf.Port

	Pods = make(map[string]*Pod, len(conf.Pods))
	Models = make(map[string]*Model, len(conf.Models))
	Prompts = make(map[string]*Prompt, len(conf.Prompts))
	Samplings = make(map[string]*Sampling, len(conf.Samplings))

	Swap = conf.Swap
	Debug = conf.Debug

	// FIXME TODO: Allow only ONE MODEL instance per ONE POD
	for id, model := range conf.Models {

		// Allow user home dir resolve with tilde ~
		path := model.Path
		if strings.HasPrefix(path, "~/") {
			usr, _ := user.Current()
			dir := usr.HomeDir
			path = filepath.Join(dir, path[2:])
		}

		// Check if the file really exists to prevent CGO panic
		if _, err := os.Stat(path); err != nil {
			Colorize("\n[magenta][ ERROR ][white] Model not found: %s\n\n", path)
			log.Infof("[ ERR ] Model not found: %s", path)
			os.Exit(0)
		}

		model.ID = id
		model.Path = path
		Models[id] = model
	}

	for id, prompt := range conf.Prompts {
		prompt.ID = id
		Prompts[id] = prompt
	}

	for id, sampling := range conf.Samplings {
		sampling.ID = id
		Samplings[id] = sampling
	}

	// -- Init all pods and models to run inside each pod - so having N * M total models ready to work

	podNum := 0
	for id, pod := range conf.Pods {

		MaxThreads += pod.Threads

		pod.ID = id
		pod.idx = podNum

		for _, layers := range pod.GPUs {
			if layers > 0 {
				pod.isGPU = true
			}
		}

		model, ok := Models[pod.Model]
		if !ok {
			Colorize("\n[magenta][ ERROR ][white] Wrong model ID in config [magenta][ %s ]\n\n", pod.Model)
			os.Exit(0)
		}

		sampling, ok := Samplings[pod.Sampling]
		if !ok {
			Colorize("\n[magenta][ ERROR ][white] Wrong sampling ID in config [magenta][ %s ]\n\n", sampling.ID)
			os.Exit(0)
		}

		gpu1 := 0
		gpu2 := 0
		gpu3 := 0
		gpu4 := 0

		if len(pod.GPUs) > 0 {
			gpu1 = pod.GPUs[0]
			if len(pod.GPUs) > 1 {
				gpu2 = pod.GPUs[1]
			}
			if len(pod.GPUs) > 2 {
				gpu3 = pod.GPUs[2]
			}
			if len(pod.GPUs) > 3 {
				gpu4 = pod.GPUs[3]
			}
		}

		ctx := C.initContext(
			C.int(podNum),
			C.CString(model.Path),
			C.int(pod.Threads),
			C.int(pod.BatchSize),
			C.int(gpu1), C.int(gpu2), C.int(gpu3), C.int(gpu4), // FIXME: Slice of GPUs
			C.int(model.ContextSize), C.int(model.Predict),
			C.int32_t(sampling.Mirostat), C.float(sampling.MirostatENT), C.float(sampling.MirostatLR),
			C.float(sampling.Temperature), C.int(sampling.TopK), C.float(sampling.TopP),
			C.float(sampling.TypicalP),
			C.float(sampling.RepetitionPenalty), C.int(sampling.PenaltyLastN),
			C.int(sampling.Janus), C.int(sampling.Depth), C.float(sampling.Scale), C.float(sampling.Hi), C.float(sampling.Lo),
			C.uint32_t(LLAMA_DEFAULT_SEED),
			C.CString(Debug),
		)

		if ctx == nil {
			Colorize("\n[magenta][ ERROR ][white] Failed to init pod for model [ %s ]\n\n", model.ID)
			os.Exit(0)
		}

		C.init(C.CString(Swap), C.CString(Debug))

		// FIXME TODO: Allow only ONE MODEL instance per ONE POD
		model.Context = ctx
		pod.model = model
		Pods[id] = pod
		podNum++
	}
}

// --- init and run Fiber server

func Run(showStatus bool) {

	app := fiber.New(fiber.Config{
		// Prefork:   true,
		Immutable: true,

		DisableStartupMessage: true,
	})

	// -- Collider API

	app.Post("/jobs/", NewJob)
	app.Delete("/jobs/:id", StopJob)
	app.Get("/jobs/status/:id", GetJobStatus)
	app.Get("/jobs/:id", GetJob)

	// -- OpenAI compatible API

	app.Post("v1/chat/completions", NewChatCompletions)

	// -- Monitoring Endpoints

	app.Get("/health", GetHealth)

	go Engine(app)

	if showStatus {
		Colorize("\n[magenta][ INIT ][light_blue] REST API running on [magenta]%s:%s", Host, Port)
	}

	log.Infof("[START] REST API running on %s:%s", Host, Port)

	err := app.Listen(Host + ":" + Port)
	if err != nil {
		Colorize("\n[magenta][ ERROR ][white] Can't start REST API on [magenta]%s:%s", Host, Port)
		log.Infof("[ ERROR ] Can't start REST API on %s:%s", Host, Port)
	}
}

// --- evergreen Engine looking for job queue and starting up to MaxPods workers

func Engine(app *fiber.App) {

	for {

		if GoShutdown && len(Queue) == 0 && RunningThreads == 0 {
			app.Shutdown()
			break
		}

		// TODO: Sync over channels
		// TODO: Some better timing + use config?
		time.Sleep(20 * time.Millisecond)

		for jobID := range Queue {

			// TODO: MaxThreads instead of MaxPods
			// FIXME: Move to outer loop?

			// simple mode with settings from CLI
			//if MaxPods > 0 && RunningPods >= MaxPods {
			//	continue
			//}

			// production mode with settings from config file
			// TODO: >= MaxThreads + pod.Model.Threads
			// TODO: Think of parallel GPU and CPU execution
			if RunningThreads >= MaxThreads {
				// continue
				break
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

			// ignore jobs placed more than [ deadline ] seconds ago
			if deadline > 0 && (now-Jobs[jobID].CreatedAt) > deadline*1000 {
				delete(Jobs, jobID)
				mu.Unlock()
				log.Infow("[ JOB ] Job was removed from queue after deadline", zap.String("jobID", jobID), zap.Int64("deadline", deadline))
				continue
			}

			var usePod *Pod
			// TODO: Implement pod priority for better serving clients
			for _, pod := range Pods {
				if !pod.isBusy {
					usePod = pod // Pods[id]
					usePod.isBusy = true
					break
				}

				// "load" the model into pod
				// WAS pod.model = Models[idx]

				// FIXME ASAP: Do we need this ?
				///// pod.model = Models[pod.Model] // TODO: more checks ( if !ok )

				//break
			}

			if usePod == nil {
				// FIXME: Something really wrong going here! We need to fix this ASAP
				// TODO: Log this case!
				mu.Unlock()
				// Colorize("\n[magenta][ INFO ][white] There no idle pods to do the job!")
				break
			}

			delete(Queue, jobID)
			Jobs[jobID].Status = "processing"

			// FIXME: Check RunningPods one more time?
			// TODO: Is it make sense to use atomic over just mutex here?
			atomic.AddInt64(&RunningPods, 1)
			atomic.AddInt64(&RunningThreads, usePod.Threads)

			mu.Unlock() // -- unlocked

			go Do(jobID, usePod) // TODO: Choose pod depending on model requested
		}
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
	Jobs[jobID].Pod = pod
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

	//model := Models[pod.Model]
	prompt, ok := Prompts[pod.Prompt]
	if !ok {
		Colorize("\n[magenta][ ERROR ][white] Error while getting prompt [magenta][ %s ]\n\n", pod.Prompt)
		os.Exit(0)
	}
	/*
		TODO: Do we need sampling here?

		sampling, ok := Samplings[pod.Sampling]
		if !ok {
			Colorize("\n[magenta][ ERROR ][white] Error while getting sampling [magenta][ %s ]\n\n", pod.Sampling)
			os.Exit(0)
		}
	*/

	// -- Inject context vars: ${DATE}, etc

	locale := monday.LocaleEnUS
	if prompt.Locale != "" {
		locale = prompt.Locale
	}

	date := monday.Format(time.Now(), "Monday 2 January 2006", monday.Locale(locale))
	date = strings.ToLower(date)
	// fmt.Printf("\nPREAMBLE BEFORE: %s", prompt.Preamble)
	preamble := strings.Replace(prompt.Preamble, "{DATE}", date, 1)
	preamble = strings.Replace(preamble, "${DATE}", date, 1) // TODO: Support just one syntax?
	// fmt.Printf("\nPREAMBLE AFTER: %s", preamble)                    // DEBUG

	// FIXME ASAP TODO - new prompt sheme
	//prompt := Jobs[jobID].Prompt
	fullPrompt := prompt.Prefix + Jobs[jobID].Prompt + prompt.Suffix
	history := Sessions[sessionID] // empty for 1) the first iteration, 2) after the limit was reached and 3) when sessions do not stored at all

	if history == "" {
		fullPrompt = preamble + fullPrompt
	} else {
		fullPrompt = history + fullPrompt
	}
	//fullPrompt = strings.Replace(fullPrompt, `\n`, "\n", -1)

	Jobs[jobID].FullPrompt = fullPrompt

	mu.Unlock() // --

	// FIXME: Do not work as expected. Empty file rise CGO exception here
	//        error loading session file: unexpectedly reached end of file
	//        do_inference: error: failed to load session file './sessions/5fb8ebd0-e0c9-4759-8f7d-35590f6c9f01'

	/*

		if _, err := os.Stat(SessionFile); err != nil {
			if os.IsNotExist(err) {
				_, err = os.Create(SessionFile)
				if err != nil {
					Colorize("\n[magenta][ ERROR ][white] Can't create session file: %s\n\n", SessionFile)
					log.Infof("[ ERROR ] Can't create session file: %s", SessionFile)
					os.Exit(0)
				}
			} else {
				Colorize("\n[magenta][ ERROR ][white] Some problems with session file: %s\n\n", SessionFile)
				log.Infof("[ ERROR ] Some problems with session file: %s", SessionFile)
				os.Exit(0)
			}
		}

	*/

	// FIXME: if model hparams were changed, session files are obsolete

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
	Jobs[jobID].Pod = nil
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
		"[ JOB ] Job was finished",
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
		// TODO: Sampling?
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
			SendString("{\n\"error\": \"service is shutting down\"\n}")
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
		Colorize("\n[magenta][ ERROR ][white] Error while parsing incoming request: [magenta]%s\n\n", err.Error())
	}

	// -- normalize prompt

	payload.Prompt = strings.Trim(payload.Prompt, "\n ")
	//payload.Mode = strings.Trim(payload.Mode, "\n ")
	//payload.Model = strings.Trim(payload.Model, "\n ")
	//payload.Translate = strings.Trim(payload.Translate, "\n ")

	// -- validate prompt

	/*

		FIXME: [ mode ] VS combination of [ prompt + model + sampling ]

		if payload.Mode != "" {
			if _, ok := Modes[payload.Mode]; !ok {
				return ctx.
					Status(fiber.StatusBadRequest).
					SendString("Wrong mode!")
			}
		}
	*/

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
			// SendString("{\n\"error\": \"wrong request ID, please use UUIDv4 format\"\n}")
			JSON(map[string]string{"error": "wrong request ID, please use UUIDv4 format"})
	}

	mu.Lock()
	if _, ok := Jobs[payload.ID]; ok {
		mu.Unlock()
		return ctx.
			Status(fiber.StatusBadRequest).
			// SendString("{\n\"error\": \"request with the same ID is already processing\"\n}")
			JSON(map[string]string{"error": "request with the same ID is already processing"})
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

	log.Infow("[JOB] New job", "jobID", payload.ID, "mode", payload.Mode, "model", payload.Model, "session", payload.Session, "prompt", payload.Prompt)

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

	if Jobs[jobID].Pod != nil {
		C.stopInference(C.int(Jobs[jobID].Pod.idx))
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

// --- POST v1/chat/completions

// {
//		"model": "gpt-3.5-turbo",
//		"messages": [
//		  {
//			"role": "system",
//			"content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."
//		  },
//		  {
//			"role": "user",
//			"content": "Compose a poem that explains the concept of recursion in programming."
//		  }
//		]
// }

func NewChatCompletions(ctx *fiber.Ctx) error {

	if GoShutdown {
		return ctx.
			Status(fiber.StatusServiceUnavailable).
			SendString("{\n\"error\": \"service is shutting down\"\n}")
	}

	type Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}

	payload := struct {
		Model    string    `json:"model,omitempty"`
		Messages []Message `json:"messages"`
	}{}

	if err := ctx.BodyParser(&payload); err != nil {
		fmt.Printf(err.Error())
		// TODO: Proper error handling
		return ctx.
			Status(fiber.StatusBadRequest).
			SendString("{\n\"error\": \"error parsing request body\n}")
	}

	// -- normalize prompt

	//payload.Prompt = strings.Trim(payload.Prompt, "\n ")
	//payload.Mode = strings.Trim(payload.Mode, "\n ")
	//payload.Model = strings.Trim(payload.Model, "\n ")
	//payload.Translate = strings.Trim(payload.Translate, "\n ")

	// -- validate prompt

	//if payload.Mode != "" {
	//	if _, ok := Modes[payload.Mode]; !ok {
	//		return ctx.
	//			Status(fiber.StatusBadRequest).
	//			SendString("Wrong mode!")
	//	}
	//}

	jobID := uuid.New().String()

	//if _, err := uuid.Parse(payload.ID); err != nil {
	//	return ctx.
	//		Status(fiber.StatusBadRequest).
	//		SendString("Wrong requerst id, please use UUIDv4 format!")
	//}

	//mu.Lock()
	//if _, ok := Jobs[id]; ok {
	//	mu.Unlock()
	//	return ctx.
	//		Status(fiber.StatusBadRequest).
	//		SendString("Request with the same id is already processing!")
	//}
	//mu.Unlock()

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
	payload.Model = ""                                          // TODO: DefaultModel
	prompt := payload.Messages[len(payload.Messages)-1].Content // TODO: Validate the last message role == "user"

	PlaceJob(jobID /*payload.Mode*/, "", payload.Model, "", prompt, "")

	log.Infow("[JOB] New job just queued", "id", jobID, "session", "", "model", payload.Model, "prompt", prompt)

	now := time.Now()
	finish := now.Add(time.Duration(deadline) * time.Second) // TODO: Change global type + naming?
	output := ""
	status := ""

	for now.Before(finish) {
		time.Sleep(1 * time.Second)

		mu.Lock() // --
		job, ok := Jobs[jobID]
		if !ok {
			// TODO: Error Handling
			// TODO: Other places too
		}
		status = job.Status
		//prompt := Jobs[jobID].Prompt
		//fullPrompt := Jobs[jobID].FullPrompt // we need the full prompt with prefix and suffix here
		output = job.Output
		//created := Jobs[jobID].CreatedAt
		//finished := Jobs[jobID].FinishedAt
		//model := Jobs[jobID].Model
		mu.Unlock() // --

		if status == "finished" {
			break
		}

		now = time.Now()
	}

	// TODO: Guard with mutex Jobs[payload.ID] access
	// TODO: Return [model] and [session] if not empty
	return ctx.JSON(fiber.Map{
		"id": jobID,
		//"session": payload.Session,
		//"model":   payload.Model,
		//"prompt": payload.Prompt,
		//"created": Jobs[payload.ID].CreatedAt,
		//"started":  Jobs[payload.ID].StartedAt,
		//"finished": Jobs[payload.ID].FinishedAt,
		//"model":    "model-xx", // TODO: Real model ID
		//"source":   "api",      // TODO: Enum for sources
		//"status": Jobs[payload.ID].Status,
		"output": output,
		"status": status,
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
