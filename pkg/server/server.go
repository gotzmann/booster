package server

// https://eli.thegreenplace.net/2019/passing-callbacks-and-pointers-to-cgo/
// https://github.com/golang/go/wiki/cgo
// https://pkg.go.dev/cmd/cgo

/*
#include <stdint.h>
void * initFromParams(char * modelName, int threads, int context, int predict, float temp, int32_t seed);
int64_t loop(void * ctx, char * jobID, char * prompt);
char * status(char * jobID);
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

type ModelConfig struct {
	ID   string // short internal name of the model
	Name string // public name for humans
	Path string // path to binary file

	Mode    string // prod, debug, ignore, etc
	Pods    int    // how many pods start
	Threads int64  // how many threads allow per pod (int64 for atomic manipulations)

	Prefix string // prompt prefix for instruct-type models
	Suffix string // prompt suffix

	Context int
	Predict int

	Temp float32

	TopK int
	TopP float32

	RepeatPenalty float32

	Mirostat    int
	MirostatTAU float32
	MirostatETA float32
}

// TODO: Logging setup
type Config struct {
	ID string // server key, should be unique within cluster

	Host string
	Port string

	Threads int // max threads (within all active pods) allowed to do compute in parallel

	AVX  bool
	NEON bool
	CUDA bool

	Default string
	Models  []ModelConfig
}

type Pod struct {
	Model   *ModelConfig   // model params
	Context unsafe.Pointer // llama.cpp context
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
	CPPMode    = 0x00
	GolangMode = 0x01
)

var (
	ServerMode int // 0 == use llama.cpp / 1 == use llama.go

	Host string
	Port string

	Vocab *ml.Vocab
	Model *llama.Model

	Ctx    *llama.Context
	Params *llama.ModelParams

	DefaultModel string // it's empty string "" for simple CLI mode and some unique key when working with configs

	// NB! All vars below are int64 to be used as atomic counters
	MaxThreads     int64 // used for PROD mode // TODO: detect hardware abilities automatically
	RunningThreads int64
	MaxPods        int64 // used for simple mode when all settings are set from CLI
	RunningPods    int64 // number of pods running at the moment - SHOULD BE int64 for atomic manipulations

	mu sync.Mutex // guards any Jobs change

	// TODO: Background watcher which will make waiting jobs obsolete after some deadline
	Jobs  map[string]*Job     // all seen jobs in any state
	Queue map[string]struct{} // queue of job IDs waiting for start

	// All pods splitted on blocks by model ID keys
	Pods     map[string][]*Pod
	IdlePods map[string][]*Pod // Indexes of idle pods
	//Contexts []unsafe.Pointer // Pointers to llama.cpp contexts
)

func init() {
	Pods = make(map[string][]*Pod, 64)      // 64 is just some commnon sense default
	IdlePods = make(map[string][]*Pod, 64)  // same here
	Jobs = make(map[string]*Job, 1024)      // 1024 is like some initial space to grow
	Queue = make(map[string]struct{}, 1024) // same here
}

// Init allocates contexts for independent pods
// TODO: Allow to load and work with different models at the same time
func Init(host string, port string, pods int, threads int, model string, context int, predict int, temp float32, seed uint32) {

	ServerMode = CPPMode
	Host = host
	Port = port
	MaxPods = int64(pods)
	RunningPods = 0
	Params.CtxSize = uint32(context)
	//IdlePods = make([]int, pods)
	//Contexts = make([]unsafe.Pointer, pods)

	// --- Starting pods incorporating isolated C++ context and runtime

	for i := 0; i < pods; i++ {
		ctx := C.initFromParams(C.CString(model), C.int(threads), C.int(context), C.int(predict), C.float(temp), C.int32_t(seed))
		if ctx == nil {
			Colorize("\n[magenta][ ERROR ][white] Failed to init pod #%d of total %d\n\n", i, pods)
			os.Exit(0)
		}
		pod := &Pod{
			Context: ctx,
			Model: &ModelConfig{
				Path:    model,
				Pods:    1,
				Threads: int64(threads),
				Context: context,
				Predict: predict,
				Temp:    temp,
				//Seed:    seed,
			},
		}

		// NB! We'll use empty string "" as key for default model
		if Pods[""] == nil {
			Pods[""] = make([]*Pod, pods)
		}
		Pods[""] = append(Pods[""], pod)
		//IdlePods[i] = i
		if IdlePods[""] == nil {
			IdlePods[""] = make([]*Pod, pods)
		}
		IdlePods[""] = append(IdlePods[""], pod)
		//Contexts[i] = ctx
	}
}

// Init allocates contexts for independent pods
// TODO: Allow to load and work with different models at the same time
func InitFromConfig(conf *Config) {

	ServerMode = CPPMode
	Host = conf.Host
	Port = conf.Port
	MaxThreads = int64(conf.Threads)
	DefaultModel = conf.Default

	for _, model := range conf.Models {

		// --- Allow user home dir resolve with tilde ~
		// TODO: // Use strings.HasPrefix so we don't match paths like "/something/~/something/"

		path := model.Path
		if strings.HasPrefix(path, "~/") {
			usr, _ := user.Current()
			dir := usr.HomeDir
			path = filepath.Join(dir, path[2:])
		}

		// FIXME: For pod = 0 .. N

		// TODO: catch panic from CGO if file not found
		ctx := C.initFromParams(
			C.CString(path),
			C.int(model.Threads),
			C.int(model.Context),
			C.int(model.Predict),
			C.float(model.Temp),
			C.int32_t( /*seed*/ -1))

		if ctx == nil {
			Colorize("\n[magenta][ ERROR ][white] Failed to init pod for model %s\n\n", model.ID)
			os.Exit(0)
		}

		pod := &Pod{
			Context: ctx,
			Model: &ModelConfig{
				ID:            model.ID,
				Name:          model.Name,
				Path:          model.Path,
				Mode:          model.Mode,
				Pods:          1,
				Threads:       model.Threads,
				Prefix:        model.Prefix,
				Suffix:        model.Suffix,
				Context:       model.Context,
				Predict:       model.Predict,
				Temp:          model.Temp,
				TopK:          model.TopK,
				TopP:          model.TopP,
				RepeatPenalty: model.RepeatPenalty,
				Mirostat:      model.Mirostat,
				MirostatTAU:   model.MirostatTAU,
				MirostatETA:   model.MirostatETA,
				//Seed:    seed,
			},
		}
		//Pods = append(Pods, pod)
		//IdlePods = append(IdlePods, pod)

		if Pods[model.ID] == nil {
			Pods[model.ID] = make([]*Pod, 0, model.Pods)
			IdlePods[model.ID] = make([]*Pod, 0, model.Pods)
		} else {
			Colorize("\n[magenta][ ERROR ][white] Model ID '%s' is not unique within config!\n\n", model.ID)
			os.Exit(0)
		}

		Pods[model.ID] = append(Pods[model.ID], pod)
		IdlePods[model.ID] = append(IdlePods[model.ID], pod)
	}

	//MaxPods = int64(pods)
	RunningPods = 0    // not needed
	RunningThreads = 0 // not needed
	//Params.CtxSize = uint32(context)
	//IdlePods = make([]int, pods)
	//Contexts = make([]unsafe.Pointer, pods)

	// --- Starting pods incorporating isolated C++ context and runtime

	//for i := 0; i < pods; i++ {
	//	ctx := C.initFromParams(C.CString(model), C.int(threads), C.int(context), C.int(predict), C.float(temp), C.int32_t(seed))
	//	if ctx == nil {
	//		Colorize("\n[magenta][ ERROR ][white] Failed to init pod #%d of total %d\n\n", i, pods)
	//		os.Exit(0)
	//	}
	//	IdlePods[i] = i
	//	Contexts[i] = ctx
	//}
}

// --- init and run Fiber server

func Run() {

	app := fiber.New(fiber.Config{
		DisableStartupMessage: true,
	})

	app.Post("/jobs/", NewJob)
	app.Get("/jobs/status/:id", GetStatus)
	app.Get("/jobs/:id", GetJob)

	go Engine()

	app.Listen(Host + ":" + Port)
}

// --- our evergreen Engine looking for job queue and starting up to MaxPods workers

func Engine() {

	for {

		// TODO: different levels of priority queues here
		for jobID := range Queue {

			// TODO: MaxThreads instead of MaxPods
			// FIXME: Move to outer loop?

			// simple mode with settings from CLI
			if MaxPods > 0 && RunningPods >= MaxPods {
				continue
			}

			// production mode with settings from config file
			// TODO: >= MaxThreads + pod.Model.Threads
			if MaxThreads > 0 && RunningThreads >= MaxThreads {
				continue
			}

			// TODO: Better to store model name right there with JobID to avoid locking
			mu.Lock()
			model := Jobs[jobID].Model
			mu.Unlock()

			if MaxThreads > 0 && len(IdlePods[model]) == 0 {
				continue
			}

			// -- move job from waiting queue to processing and assign it pod from idle pool
			// TODO: Use different mutexes for Jobs map, Pods map and maybe for atomic counters

			mu.Lock()
			delete(Queue, jobID)
			Jobs[jobID].Status = "processing"
			//if model == "" {
			//	model = DefaultModel
			//	Jobs[jobID].Model = model
			//}
			pod := IdlePods[model][len(IdlePods[model])-1]
			if pod == nil {
				// TODO: Something really wrong going here! We need to fix this ASAP
				// TODO: Log this case!
				mu.Unlock()
				Colorize("\n[magenta][ ERROR ][white] Failed to get idle pod for '%s' model!\n\n", model)
				continue
			}
			IdlePods[model] = IdlePods[model][:len(IdlePods[model])-1]
			mu.Unlock()

			// FIXME: Check RunningPods one more time?
			// TODO: Is it make sense to use atomic over just mutex here?
			atomic.AddInt64(&RunningPods, 1)
			atomic.AddInt64(&RunningThreads, pod.Model.Threads)

			go Do(jobID, pod)
		}

		// TODO: Sync over channels
		time.Sleep(100 * time.Millisecond)
	}
}

// --- worker doing the "job" of transforming boring prompt into magic output

func Do(jobID string, pod *Pod) {

	defer atomic.AddInt64(&RunningPods, -1)
	defer atomic.AddInt64(&RunningThreads, -pod.Model.Threads)
	defer runtime.GC()

	// TODO: Proper logging
	// fmt.Printf("\n[ PROCESSING ] Starting job # %s", jobID)

	mu.Lock()
	Jobs[jobID].StartedAt = time.Now().Unix()
	//Jobs[jobID].Timings = make([]int64, 0, 1024) // Reserve reasonable space (like context size) for storing token evaluation timings
	// TODO: Play with prompt without leading space
	//prompt := " " + Jobs[jobID].Prompt // add a space to match LLaMA tokenizer behavior
	// TODO: Allow setting prefix/suffix from CLI
	// TODO: Implement translation for prompt elsewhere
	prompt := " " + pod.Model.Prefix + Jobs[jobID].Prompt + pod.Model.Suffix // add a space to match LLaMA tokenizer behavior
	mu.Unlock()

	if ServerMode == CPPMode { // --- use llama.cpp backend

		//tokenCount := C.loop(Contexts[pod], C.CString(jobID), C.CString(prompt))
		tokenCount := C.loop(pod.Context, C.CString(jobID), C.CString(prompt))

		// TODO: Trim prompt from beginning
		result := C.GoString(C.status(C.CString(jobID)))

		// TODO: Move some slow ops outside of critical section
		mu.Lock()
		Jobs[jobID].FinishedAt = time.Now().Unix()
		Jobs[jobID].Output = strings.Trim(result, "\n ")
		Jobs[jobID].Status = "finished"
		Jobs[jobID].TokenCount = int64(tokenCount)
		Jobs[jobID].TokenEval = int64(C.timing(C.CString(jobID)))
		IdlePods[pod.Model.ID] = append(IdlePods[pod.Model.ID], pod) // return pod to the pool
		mu.Unlock()

	} else { // --- use llama.go framework

		// tokenize the prompt
		embdPrompt := ml.Tokenize(Vocab, prompt, true)

		// ring buffer for last N tokens
		lastNTokens := ring.New(int(Params.CtxSize))

		// method to append a token to the ring buffer
		appendToken := func(token uint32) {
			lastNTokens.Value = token
			lastNTokens = lastNTokens.Next()
		}

		// zeroing the ring buffer
		for i := 0; i < int(Params.CtxSize); i++ {
			appendToken(0)
		}

		evalCounter := 0
		tokenCounter := 0
		pastCount := uint32(0)
		consumedCount := uint32(0)           // number of tokens, already processed from the user prompt
		remainedCount := Params.PredictCount // how many tokens we still need to generate to achieve predictCount
		embd := make([]uint32, 0, Params.BatchSize)

		evalPerformance := make([]int64, 0, Params.PredictCount)
		samplePerformance := make([]int64, 0, Params.PredictCount)
		fullPerformance := make([]int64, 0, Params.PredictCount)

		// new context opens sync channel and starts workers for tensor compute
		ctx := llama.NewContext(Model, Params)

		for remainedCount > 0 {

			// TODO: Store total time of evaluation and average per token + token count
			start := time.Now().UnixNano()

			if len(embd) > 0 {

				// infinite text generation via context swapping
				// if we run out of context:
				// - take the n_keep first tokens from the original prompt (via n_past)
				// - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch

				if pastCount+uint32(len(embd)) > Params.CtxSize {
					leftCount := pastCount - Params.KeepCount
					pastCount = Params.KeepCount

					// insert n_left/2 tokens at the start of embd from last_n_tokens
					// embd = append(lastNTokens[:leftCount/2], embd...)
					embd = append(llama.ExtractTokens(lastNTokens.Move(-int(leftCount/2)), int(leftCount/2)), embd...)
				}

				evalStart := time.Now().UnixNano()
				if err := llama.Eval(ctx, Vocab, Model, embd, pastCount, Params); err != nil {
					// TODO: Finish job properly with [failed] status
				}
				evalPerformance = append(evalPerformance, time.Now().UnixNano()-evalStart)
				evalCounter++
			}

			pastCount += uint32(len(embd))
			embd = embd[:0]

			if int(consumedCount) < len(embdPrompt) {

				for len(embdPrompt) > int(consumedCount) && len(embd) < int(Params.BatchSize) {

					embd = append(embd, embdPrompt[consumedCount])
					appendToken(embdPrompt[consumedCount])
					consumedCount++
				}

			} else {

				//if Params.IgnoreEOS {
				//	Ctx.Logits[ml.TOKEN_EOS] = 0
				//}

				sampleStart := time.Now().UnixNano()
				id := llama.SampleTopPTopK( /*ctx,*/ ctx.Logits,
					lastNTokens, Params.RepeatLastN,
					Params.TopK, Params.TopP,
					Params.Temp, Params.RepeatPenalty)
				samplePerformance = append(samplePerformance, time.Now().UnixNano()-sampleStart)

				appendToken(id)

				// replace end of text token with newline token when in interactive mode
				//if id == ml.TOKEN_EOS && Params.Interactive && !Params.Instruct {
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
				token := ml.Token2Str(Vocab, id) // TODO: Simplify

				mu.Lock()
				Jobs[jobID].Output += token
				mu.Unlock()
			}
		}

		// close sync channel and stop compute workers
		ctx.ReleaseContext()

		mu.Lock()
		Jobs[jobID].FinishedAt = time.Now().Unix()
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

	timing := time.Now().Unix()

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

	payload := struct {
		ID      string `json:"id"`
		Session string `json:"session"`
		Model   string `json:"model"`
		Prompt  string `json:"prompt"`
	}{}

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

	// TODO: Tokenize and check for max tokens properly
	if len(payload.Prompt) >= int(Params.CtxSize)*3 {
		return ctx.
			Status(fiber.StatusBadRequest).
			SendString(fmt.Sprintf("Prompt length is more than allowed %d tokens!", Params.CtxSize))
	}

	if payload.Model != "" {
		if _, ok := Pods[payload.Model]; !ok {
			return ctx.
				Status(fiber.StatusBadRequest).
				SendString(fmt.Sprintf("Model with name '%s' is not found!", payload.Model))
		}
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
		"prompt": payload.Prompt,
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
		"prompt":   Jobs[id].Prompt,
		"output":   Jobs[id].Output,
		"created":  Jobs[id].CreatedAt,
		"started":  Jobs[id].StartedAt,
		"finished": Jobs[id].FinishedAt,
		"model":    "model-xx", // TODO: Real model ID
		"status":   Jobs[id].Status,
	})
}

// Colorize is a wrapper for colorstring.Color() and fmt.Fprintf()
// Join colorstring and go-colorable to allow colors both on Mac and Windows
// TODO: Implement as a small library
func Colorize(format string, opts ...interface{}) (n int, err error) {
	var DefaultOutput = colorable.NewColorableStdout()
	return fmt.Fprintf(DefaultOutput, colorstring.Color(format), opts...)
}
