package server

// https://eli.thegreenplace.net/2019/passing-callbacks-and-pointers-to-cgo/
// https://github.com/golang/go/wiki/cgo
// https://pkg.go.dev/cmd/cgo

/*
#include <stdint.h>
void * initContext(
	int idx,
	char * modelName,
	int threads, int gpuLayers,
	int context, int predict,
	int mirostat, float mirostat_tau, float mirostat_eta,
	float temp, int topK, float topP,
	float repeat_penalty, int repeat_last_n,
	int32_t seed);
int64_t loop(int idx, void * ctx, char * jobID, char * prompt);
const char * status(char * jobID);
int64_t timing(char * jobID);
*/
import "C"

import (
	"container/ring"
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
	"github.com/google/uuid"
	colorable "github.com/mattn/go-colorable"
	"github.com/mitchellh/colorstring"

	"github.com/gotzmann/llamazoo/pkg/llama"
	"github.com/gotzmann/llamazoo/pkg/ml"
)

// TODO: Check the host:port is free before starting listening

// TODO: Helicopter View - how to work with balancers and multi-pod architectures?
// TODO: Rate Limiter based on end-user IP address
// TODO: Guard access with API Tokens
// TODO: Each use of C.CString() should be complemented with C.free() operation
// TODO: GetStatus - update partial output if processing within C++ core
// TODO: Looks like all generations use THE SAME seed?

// Unix timestamps VS ISO-8601 Stripe perspective:
// https://dev.to/stripe/how-stripe-designs-for-dates-and-times-in-the-api-3eoh
// TODO: UUID vs string for job ID
// TODO: Unix timestamp vs ISO for date and time

type Model struct {
	ID   string // short internal name of the model
	Name string // public name for humans
	Path string // path to binary file

	Context unsafe.Pointer // *llama.Context

	Prefix string // prompt prefix for instruct-type models
	Suffix string // prompt suffix

	ContextSize int
	Predict     int

	Mirostat    int
	MirostatTAU float32
	MirostatETA float32

	Temp float32
	TopK int
	TopP float32

	RepeatPenalty float32
	RepeatLastN   int
}

// TODO: Logging setup
type Config struct {
	ID string // server key, should be unique within cluster

	Host string
	Port string

	AVX  bool
	NEON bool
	CUDA bool

	Pods      int     // pods count
	Threads   []int64 // threads count for each pod
	GPUs      []int64 // GPU number selector
	GPULayers []int   // how many layers offload to Apple GPU?

	Models []Model

	DefaultModel string // default model ID
}

type Pod struct {
	idx       int    // pod index
	isBusy    bool   // do pod instance doing some job?
	Threads   int64  // how many threads in use
	GPU       int64  // GPU index
	GPULayers int64  // how many layers offload to Apple GPU?
	Model     *Model // model params
}

type Job struct {
	ID      string
	Session string // ID of continuous user session in chat mode
	Status  string
	Prompt  string
	Output  string

	CreatedAt  int64
	StartedAt  int64
	FinishedAt int64

	Seed  int64  // TODO: Store seed
	Model string // TODO: Store model ID, "" for default should be then replaced

	TokenCount int64 // total tokens processed (icnluding prompt)
	TokenEval  int64 // timing per token (prompt + output), ms
}

const (
	LLAMA_CPP = 0x00
	LLAMA_GO  = 0x01
	EXLLAMA   = 0x02
)

var (
	ServerMode int // LLAMA_CPP by default

	Host string
	Port string

	GoShutdown bool // signal the service should go graceful shutdown

	// Data for running one model from CLI without pods instantiating
	vocab  *ml.Vocab
	model  *llama.Model
	ctx    *llama.Context
	params *llama.ModelParams

	DefaultModel string // it's empty string "" for simple CLI mode and some unique key when working with configs

	// NB! All vars below are int64 to be used as atomic counters
	MaxThreads     int64 // used for PROD mode // TODO: detect hardware abilities automatically
	RunningThreads int64
	RunningPods    int64 // number of pods running at the moment - SHOULD BE int64 for atomic manipulations

	mu sync.Mutex // guards any Jobs change

	// TODO: Background watcher which will make waiting jobs obsolete after some deadline
	Jobs  map[string]*Job     // all seen jobs in any state
	Queue map[string]struct{} // queue of job IDs waiting for start

	Pods   []*Pod              // There N pods with some threads within as described in config
	Models map[string][]*Model // Each unique model identified by key has N instances ready to run in pods
)

func init() {
	Jobs = make(map[string]*Job, 1024)      // 1024 is like some initial space to grow
	Queue = make(map[string]struct{}, 1024) // same here

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
	pods int, threads, gpuLayers int64,
	model, prefix, suffix string,
	context, predict int,
	mirostat int, mirostatTAU float32, mirostatETA float32,
	temp float32, topK int, topP float32,
	repeatPenalty float32, repeatLastN int,
	seed uint32) {

	ServerMode = LLAMA_CPP
	Host = host
	Port = port
	RunningPods = 0
	params.CtxSize = uint32(context)
	Pods = make([]*Pod, pods)
	Models = make(map[string][]*Model)

	// model from CLI will have empty name by default
	if _, ok := Models[""]; !ok {
		Models[""] = make([]*Model, pods)
	}

	// --- Starting pods incorporating isolated C++ context and runtime

	for pod := 0; pod < pods; pod++ {

		MaxThreads += threads
		Pods[pod] = &Pod{
			idx:     pod,
			Threads: threads,
		}

		// Check if file exists to prevent CGO panic
		if _, err := os.Stat(model); err != nil {
			Colorize("\n[magenta][ ERROR ][white] Model file not found: %s\n\n", model)
			os.Exit(0)
		}

		ctx := C.initContext(
			C.int(pod),
			C.CString(model),
			C.int(threads), C.int(gpuLayers),
			C.int(context), C.int(predict),
			C.int(mirostat), C.float(mirostatTAU), C.float(mirostatETA),
			C.float(temp), C.int(topK), C.float(topP),
			C.float(repeatPenalty), C.int(repeatLastN),
			C.int32_t(seed))

		if ctx == nil {
			Colorize("\n[magenta][ ERROR ][white] Failed to init pod #%d of total %d\n\n", pod, pods)
			os.Exit(0)
		}

		Models[""][pod] = &Model{
			Path:        model,
			Context:     ctx,
			Prefix:      prefix,
			Suffix:      suffix,
			ContextSize: context,
			Predict:     predict,

			Mirostat:    mirostat,
			MirostatTAU: mirostatTAU,
			MirostatETA: mirostatETA,

			Temp: temp,
			TopK: topK,
			TopP: topP,

			RepeatPenalty: repeatPenalty,
			RepeatLastN:   repeatLastN,
		}
	}
}

// Init allocates contexts for independent pods
func InitFromConfig(conf *Config) {

	// -- some validations TODO: move to better place

	if conf.Pods != len(conf.Threads) {
		Colorize("\n[magenta][ ERROR ][white] Please fix config! Treads array should have numbers for each pod of total %d\n\n", conf.Pods)
		os.Exit(0)
	}

	for conf.Pods != len(conf.GPULayers) {
		Colorize("\n[magenta][ ERROR ][white] Please fix config! Set number of GPU layers for each pod of total %d\n\n", conf.Pods)
		os.Exit(0)
	}

	defaultModelFound := false
	for _, model := range conf.Models {
		if conf.DefaultModel == model.ID {
			defaultModelFound = true
		}
	}

	if !defaultModelFound {
		Colorize("\n[magenta][ ERROR ][white] Default model not found: %s\n\n", conf.DefaultModel)
		os.Exit(0)
	}

	// -- init golbal settings

	ServerMode = LLAMA_CPP
	Host = conf.Host
	Port = conf.Port
	DefaultModel = conf.DefaultModel
	Pods = make([]*Pod, conf.Pods)
	Models = make(map[string][]*Model)

	// -- Init all pods and models to run inside each pod - so having N * M total models ready to work

	for pod, threads := range conf.Threads {

		MaxThreads += threads
		Pods[pod] = &Pod{
			idx:     pod,
			Threads: threads,
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
				Colorize("\n[magenta][ ERROR ][white] Model file not found: %s\n\n", path)
				os.Exit(0)
			}

			// TODO: catch panic from CGO if file not found
			ctx := C.initContext(
				C.int(pod),
				C.CString(path),
				C.int(threads), C.int(conf.GPULayers[pod]),
				C.int(model.ContextSize), C.int(model.Predict),
				C.int(model.Mirostat), C.float(model.MirostatTAU), C.float(model.MirostatETA),
				C.float(model.Temp), C.int(model.TopK), C.float(model.TopP),
				C.float(model.RepeatPenalty), C.int(model.RepeatLastN),
				C.int32_t(-1))

			if ctx == nil {
				Colorize("\n[magenta][ ERROR ][white] Failed to init pod for model %s\n\n", model.ID)
				os.Exit(0)
			}

			// Each model might be running an all pods, thus need to have N*M contexts available
			if _, ok := Models[model.ID]; !ok {
				Models[model.ID] = make([]*Model, conf.Pods)
			}

			Models[model.ID][pod] = &Model{
				ID:      model.ID,
				Name:    model.Name,
				Path:    model.Path,
				Context: ctx,

				Prefix: model.Prefix,
				Suffix: model.Suffix,

				ContextSize: model.ContextSize,
				Predict:     model.Predict,

				Mirostat:    model.Mirostat,
				MirostatTAU: model.MirostatTAU,
				MirostatETA: model.MirostatETA,

				Temp: model.Temp,
				TopK: model.TopK,
				TopP: model.TopP,

				RepeatPenalty: model.RepeatPenalty,
				RepeatLastN:   model.RepeatLastN,
			}
		}
	}
}

// --- init and run Fiber server

func Run() {

	app := fiber.New(fiber.Config{
		DisableStartupMessage: true,
	})

	app.Post("/jobs/", NewJob)
	app.Get("/jobs/status/:id", GetStatus)
	app.Get("/jobs/:id", GetJob)

	go Engine(app)

	app.Listen(Host + ":" + Port)
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
			if now-Jobs[jobID].CreatedAt > 3*60*1000 {
				delete(Jobs, jobID)
				mu.Unlock()
				continue
			}

			Jobs[jobID].Status = "processing"
			//if model == "" {
			//	model = DefaultModel
			//	Jobs[jobID].Model = model
			//}

			// TODO: replace len(Pods) for defined value
			var pod *Pod
			var idx int
			//for i := 0; i < len(Pods); i++ {
			for idx, pod = range Pods {
				if pod.isBusy {
					continue
				}
				pod.isBusy = true
				// "load" the model into pod
				model := Jobs[jobID].Model
				pod.Model = Models[model][idx]
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

	mu.Lock()
	Jobs[jobID].StartedAt = time.Now().UnixMilli()
	//Jobs[jobID].Timings = make([]int64, 0, 1024) // Reserve reasonable space (like context size) for storing token evaluation timings
	// TODO: Play with prompt without leading space
	//prompt := " " + Jobs[jobID].Prompt // add a space to match LLaMA tokenizer behavior
	// TODO: Allow setting prefix/suffix from CLI
	// TODO: Implement translation for prompt elsewhere
	prompt := " " + pod.Model.Prefix + Jobs[jobID].Prompt + pod.Model.Suffix // add a space to match LLaMA tokenizer behavior
	mu.Unlock()

	if ServerMode == LLAMA_CPP { // --- use llama.cpp backend

		//tokenCount := C.loop(Contexts[pod], C.CString(jobID), C.CString(prompt))
		tokenCount := C.loop(C.int(pod.idx), pod.Model.Context, C.CString(jobID), C.CString(prompt))

		// TODO: Trim prompt from beginning
		result := C.GoString(C.status(C.CString(jobID)))

		// TODO: Move some slow ops outside of critical section
		mu.Lock()
		Jobs[jobID].FinishedAt = time.Now().UnixMilli()
		Jobs[jobID].Status = "finished"
		Jobs[jobID].TokenCount = int64(tokenCount)
		Jobs[jobID].TokenEval = int64(C.timing(C.CString(jobID)))

		// -- remove suffix and prefix from the output
		// TODO: Better processing here

		result = strings.Trim(result, "\n ")
		prompt = strings.Trim(prompt, "\n ") // TODO: Extra step - not needed, just dont use leading space
		if strings.HasPrefix(result, prompt) {
			result = result[len(prompt):]
		}
		Jobs[jobID].Output = strings.Trim(result, "\n ")

		/////IdlePods[pod.Model.ID] = append(IdlePods[pod.Model.ID], pod) // return pod to the pool
		pod.isBusy = false
		mu.Unlock()

	} else { // --- use llama.go framework

		// tokenize the prompt
		embdPrompt := ml.Tokenize(vocab, prompt, true)

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

					// insert n_left/2 tokens at the start of embd from last_n_tokens
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
				id := llama.SampleTopPTopK( /*ctx,*/ ctx.Logits,
					lastNTokens, params.RepeatLastN,
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
}

// --- Place new job into queue

func PlaceJob(jobID, model, session, prompt string) {

	timing := time.Now().UnixMilli()

	mu.Lock()

	Jobs[jobID] = &Job{
		ID:        jobID,
		Model:     model,
		Session:   session,
		Prompt:    prompt,
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
//	    "prompt": "Why Golang is so popular?"
//	}

func NewJob(ctx *fiber.Ctx) error {

	if GoShutdown {
		return ctx.
			Status(fiber.StatusServiceUnavailable).
			SendString("Service shutting down...")
	}

	payload := struct {
		ID      string `json:"id"`
		Session string `json:"session"`
		Model   string `json:"model"`
		Prompt  string `json:"prompt"`
	}{}

	// normalize prompt
	payload.Prompt = strings.Trim(payload.Prompt, "\n ")
	payload.Model = strings.Trim(payload.Model, "\n ")

	if payload.Model != "" {
		if _, ok := Models[payload.Model]; !ok {
			return ctx.
				Status(fiber.StatusBadRequest).
				SendString("Wrong model name!")
		}
	}

	if err := ctx.BodyParser(&payload); err != nil {
		// TODO: Proper error handling
	}

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

	if payload.Model != "" {
		// FIXME: Refactor ASAP
		/////if _, ok := Pods[payload.Model]; !ok {
		/////	return ctx.
		/////		Status(fiber.StatusBadRequest).
		/////		SendString(fmt.Sprintf("Model with name '%s' is not found!", payload.Model))
		/////}
	} else {
		payload.Model = DefaultModel
	}

	PlaceJob(payload.ID, payload.Model, payload.Session, payload.Prompt)

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

// --- GET /jobs/status/:id

func GetStatus(ctx *fiber.Ctx) error {

	id := ctx.Params("id")

	if _, err := uuid.Parse(id); err != nil {
		return ctx.
			Status(fiber.StatusBadRequest).
			SendString("Wrong UUID4 id for request!")
	}

	if _, ok := Jobs[id]; !ok {
		return ctx.
			Status(fiber.StatusBadRequest).
			SendString("Request ID was not found!")
	}

	// TODO: Guard with mutex
	return ctx.JSON(fiber.Map{
		"status": Jobs[id].Status,
	})
}

// --- GET /jobs/:id

func GetJob(ctx *fiber.Ctx) error {

	id := ctx.Params("id")

	if _, err := uuid.Parse(id); err != nil {
		return ctx.
			Status(fiber.StatusBadRequest).
			SendString("Wrong UUID4 id for request!")
	}

	if _, ok := Jobs[id]; !ok {
		return ctx.
			Status(fiber.StatusBadRequest).
			SendString("Request ID was not found!")
	}

	// TODO: Guard with mutex
	return ctx.JSON(fiber.Map{
		"id":       id,
		"status":   Jobs[id].Status,
		"prompt":   Jobs[id].Prompt,
		"output":   Jobs[id].Output,
		"created":  Jobs[id].CreatedAt,
		"started":  Jobs[id].StartedAt,
		"finished": Jobs[id].FinishedAt,
		//"model":    "model-xx", // TODO: Real model ID
	})
}

// Colorize is a wrapper for colorstring.Color() and fmt.Fprintf()
// Join colorstring and go-colorable to allow colors both on Mac and Windows
// TODO: Implement as a small library
func Colorize(format string, opts ...interface{}) (n int, err error) {
	var DefaultOutput = colorable.NewColorableStdout()
	return fmt.Fprintf(DefaultOutput, colorstring.Color(format), opts...)
}
