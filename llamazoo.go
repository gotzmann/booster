package main

// TODO: Use UUID instead of string https://github.com/google/uuid/blob/master/uuid.go
// TODO: Benchmark map[string] vs map[UUID] by memory and performance for accessing 1 million elements
// TODO: Option to disable params.use_mmap
// TODO: Replace [ END ] token with some UTF visual sign (end of the paragraph, etc.)
// TODO: Read mirostat paper https://arxiv.org/pdf/2007.14966.pdf
// TODO: Support instruct prompts for Vicuna and other
// TODO: model = 13B/ggml-model-q4_0.bin + TopK = 40 + seed = 1683553932 => Why Golang is not so popular in Pakistan?
// TODO: TopP and TopK as CLI parameters
// Perplexity graph for different models https://github.com/ggerganov/llama.cpp/pull/1004
// Yet another graph, LLaMA 7B, 30B, 65B | 4Q | F16  https://github.com/ggerganov/llama.cpp/pull/835
// Read about quantization and perplexity experiments https://github.com/saharNooby/rwkv.cpp/issues/12
// wiki-raw datasets https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/
// Perplexity for all models https://github.com/ggerganov/llama.cpp/discussions/406
// GPTQ vs RTN Perplexity https://github.com/qwopqwop200/GPTQ-for-LLaMa

// https://kofo.dev/build-tags-in-golang

// invalid flag in #cgo CFLAGS: -mfma -mf16c
// argument unused during compilation: -mavx -mavx2  -msse3

// find / -name vector 2>/dev/null

// void * initFromParams(char * modelName, int threads);
// void doInference(void * ctx, char * jobID, char * prompt);
// const char * status(char * jobID);

// #cgo LDFLAGS: bridge.o ggml.o llama.o -lstdc++ -framework Accelerate
// cgo darwin LDFLAGS: bridge.o ggml.o llama.o k_quants.o ggml-metal.o -lstdc++ -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders

/*
const char * status(char * jobID);
#cgo CFLAGS:   -I. -O3 -fPIC -pthread -std=c17 -DNDEBUG -DGGML_USE_METAL -DGGML_METAL_NDEBUG
#cgo CXXFLAGS: -I. -O3 -fPIC -pthread -std=c++17 -DNDEBUG -DGGML_USE_METAL
#cgo linux LDFLAGS: bridge.o ggml.o llama.o k_quants.o -lstdc++ -lm
#cgo darwin LDFLAGS: bridge.o ggml.o llama.o k_quants.o ggml-metal.o -lstdc++ -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
*/
import "C"

import (
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	config "github.com/golobby/config/v3"
	"github.com/golobby/config/v3/pkg/feeder"
	flags "github.com/jessevdk/go-flags"
	colorable "github.com/mattn/go-colorable"
	"github.com/mitchellh/colorstring"
	"github.com/pkg/profile"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/gotzmann/llamazoo/pkg/server"
)

const VERSION = "0.9.9"

type Options struct {
	Prompt        string  `long:"prompt" description:"Text prompt from user to feed the model input"`
	Model         string  `long:"model" description:"Path and file name of converted .bin LLaMA model [ llama-7b-fp32.bin, etc ]"`
	Prefix        string  `long:"prefix" description:"Prompt prefix if needed, like \"### Instruction:\""`
	Suffix        string  `long:"suffix" description:"Prompt suffix if needed, like \"### Response:\""`
	Seed          uint32  `long:"seed" description:"Seed number for random generator initialization [ current Unix time by default ]"`
	Server        bool    `long:"server" description:"Start in Server Mode acting as REST API endpoint"`
	Debug         bool    `long:"debug" description:"Stream debug info to console while processing requests"`
	Log           string  `long:"log" description:"Log file location to save all events in Server mode"`
	Deadline      int64   `long:"deadline" description:"Time in seconds after which unprocessed jobs will be deleted from the queue"`
	Host          string  `long:"host" description:"Host to allow requests from in Server mode [ localhost by default ]"`
	Port          string  `long:"port" description:"Port listen to in Server Mode [ 8080 by default ]"`
	Pods          int     `long:"pods" description:"Maximum pods of parallel execution allowed in Server mode [ 1 by default ]"`
	Threads       int64   `long:"threads" description:"Max number of CPU cores you allow to use for one pod [ all cores by default ]"`
	Context       uint32  `long:"context" description:"Context size in tokens [ 1024 by default ]"`
	Predict       uint32  `long:"predict" description:"Number of tokens to predict [ 512 by default ]"`
	Mirostat      int     `long:"mirostat" description:"Mirostat version [ zero or disabled by default ]"`
	MirostatTAU   float32 `long:"mirostat-tau" description:"Mirostat TAU value [ 0.1 by default ]"`
	MirostatETA   float32 `long:"mirostat-eta" description:"Mirostat ETA value [ 0.1 by default ]"`
	Temp          float32 `long:"temp" description:"Model temperature hyper parameter [ 0.4 by default ]"`
	TopK          int     `long:"top-k" description:"TopK parameter for the model [ 8 by default ]"`
	TopP          float32 `long:"top-p" description:"TopP parameter for the model [ 0.8 by default ]"`
	RepeatPenalty float32 `long:"repeat-penalty" description:"RepeatPenalty [ 1.1 by default ]"`
	RepeatLastN   int     `long:"repeat-last-n" description:"RepeatLastN [ -1 by default ]"`
	Silent        bool    `long:"silent" description:"Hide welcome logo and other output [ shown by default ]"`
	Chat          bool    `long:"chat" description:"Chat with user in interactive mode instead of compute over static prompt"`
	Dir           string  `long:"dir" description:"Directory used to download .bin model specified with --model parameter [ current by default ]"`
	Profile       bool    `long:"profile" description:"Profe CPU performance while running and store results to cpu.pprof file"`
	GPULayers     int64   `long:"gpu-layers" description:"Use Apple GPU inference, offload NN layers"`
	UseAVX        bool    `long:"avx" description:"Enable x64 AVX2 optimizations for Intel and AMD machines"`
	UseNEON       bool    `long:"neon" description:"Enable ARM NEON optimizations for Apple and ARM machines"`
	Ignore        bool    `long:"ignore" description:"Ignore server JSON and YAML configs, use only CLI params"`
}

var (
	doPrint bool = true
	doLog   bool = false
	conf    server.Config
)

func main() {

	// --- parse command line options

	opts := parseOptions()

	// --- read config from JSON or YAML

	var feed config.Feeder
	if !opts.Ignore {

		if _, err := os.Stat("config.json"); err == nil {
			feed = feeder.Json{Path: "config.json"}
		} else if _, err := os.Stat("config.yaml"); err == nil {
			feed = feeder.Yaml{Path: "config.yaml"}
		}

		if feed != nil {
			err := config.New().AddFeeder(feed).AddStruct(&conf).Feed()
			if err != nil {
				Colorize("\n[magenta][ ERROR ][white] Can't parse config from JSON file! %s\n\n", err.Error())
				os.Exit(0)
			}
		}
	}

	if opts.Profile {
		defer profile.Start(profile.ProfilePath(".")).Stop()
	}

	var zapWriter zapcore.WriteSyncer
	zapConfig := zap.NewProductionEncoderConfig()
	zapConfig.NameKey = "llamazoo" // TODO: pod name from config?
	//zapConfig.CallerKey = ""       // do not log caller like "llamazoo/llamazoo.go:156"
	zapConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	fileEncoder := zapcore.NewJSONEncoder(zapConfig)
	if opts.Log != "" {
		conf.Log = opts.Log
	}
	if conf.Log != "" {
		logFile, err := os.OpenFile(conf.Log, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			Colorize("\n[light_magenta][ ERROR ][white] Can't init logging, shutdown...\n\n")
			os.Exit(0)
		}
		zapWriter = zapcore.AddSync(logFile)
		//defaultLogLevel := zapcore.DebugLevel
	} else {
		zapWriter = os.Stderr
	}
	core := zapcore.NewTee(zapcore.NewCore(fileEncoder, zapWriter, zapcore.DebugLevel))
	//logger := zap.New(core, zap.AddCaller(), zap.AddStacktrace(zapcore.ErrorLevel))
	logger := zap.New(core)
	log := logger.Sugar()

	if !opts.Server {
		showLogo()
	} else {
		log.Infof("[START] LLaMAZoo v%s starting...", VERSION)
	}

	// --- Allow graceful shutdown via OS signals
	// https://ieftimov.com/posts/four-steps-daemonize-your-golang-programs/

	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM)

	// --- Do all we need in case of graceful shutdown or unexpected panic

	defer func() {
		signal.Stop(signalChan)
		logger.Sync()
		reason := recover()
		if reason != nil {
			Colorize("\n[light_magenta][ ERROR ][white] %s\n\n", reason)
			log.Error("%s", reason)
			os.Exit(0)
		}
		Colorize("\n[light_magenta][ STOP ][light_blue] LLaMAZoo was stopped. Arrivederci!\n\n")
		log.Info("[STOP] LLaMAZoo was stopped. Arrivederci!")
	}()

	// --- Listen for OS signals in background

	go func() {
		select {
		case <-signalChan:

			// -- break execution immediate when DEBUG

			if opts.Debug {
				Colorize("\n[light_magenta][ STOP ][light_blue] Immediate shutdown...\n\n")
				log.Info("[STOP] Immediate shutdown...")
				os.Exit(0)
			}

			// -- wait while job will be done otherwise

			server.GoShutdown = true
			Colorize("\n[light_magenta][ STOP ][light_blue] Graceful shutdown...")
			log.Info("[STOP] Graceful shutdown...")
			pending := len(server.Queue)
			if pending > 0 {
				pending += conf.Pods
				Colorize("\n[light_magenta][ STOP ][light_blue] Wait while [light_magenta][ %d ][light_blue] requests will be finished...", pending)
				log.Infof("[STOP] Wait while [ %d ] requests will be finished...", pending)
			}
		}
	}()

	// -- DEBUG

	// ==== 7B ====

	// https://huggingface.co/eachadea/ggml-vicuna-7b-1.1
	//opts.Model = "/Users/me/models/7B/ggml-vic7b-q4_0.bin" // censored (with mirostat)

	// https://huggingface.co/eachadea/ggml-wizardlm-7b/tree/main
	//opts.Model = "/Users/me/models/7B/ggml-wizardlm-7b-q8_0.bin" // perfect with mirostat v2 (censored with mirostat)

	// -- v2

	// https://huggingface.co/TheBloke/wizardLM-7B-GGML/tree/main
	//opts.Model = "/Users/me/models/7B/wizardLM-7b.ggml.q4_0.bin" // v2 - THAT's THE MODEL! Sooo great but censored | As an AI language model...
	//opts.Model = "/Users/me/models/7B/wizardLM-7b.ggml.q5_0.bin"
	//opts.Model = "/Users/me/models/7B/wizardLM-7b.ggml.q5_1.bin" //
	//opts.Model = "/Users/me/models/7B/wizardLM-7b-uncensored.ggml.q4_0.bin"

	// https://huggingface.co/TheBloke/koala-7B-GGML/tree/main
	//opts.Model = "/Users/me/models/7B/koala-7B.ggml.q4_0.bin" // v2 - bad?

	// -- v3

	//opts.Model = "/Users/me/models/7B/llama-7b-ggml-v3-q4_0.bin"

	// https://huggingface.co/TheBloke/wizardLM-7B-GGML/tree/main
	//opts.Model = "/Users/me/models/7B/wizardLM-7B.ggmlv3.q4_0.bin" // GOOD not GREAT! TopK for logic
	//opts.Model = "/Users/me/models/7B/wizardLM-7B.ggmlv3.q5_0.bin" // 50% slower than 4bit, no quality boost at all

	// https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GGML/tree/main
	//opts.Model = "/Users/me/models/7B/WizardLM-7B-uncensored.ggmlv3.q4_0.bin"
	//opts.Model = "/Users/me/models/7B/WizardLM-7B-uncensored.ggmlv3.q5_0.bin"

	// https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GGML/tree/main
	//opts.Model = "/Users/me/models/7B/Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_0.bin"

	// https://huggingface.co/jondurbin/airoboros-7b-ggml-q4_0/tree/main
	//opts.Model = "/Users/me/models/7B/airoboros-7b-ggml-q4_0.bin"

	// https://huggingface.co/TheBloke/guanaco-7B-GGML/tree/main
	//opts.Model = "/Users/me/models/7B/guanaco-7B.ggmlv3.q4_0.bin" // 50/50
	//opts.Model = "/Users/me/models/7B/guanaco-7B.ggmlv3.q4_1.bin" // 50/50

	// https://huggingface.co/TheBloke/Project-Baize-v2-7B-GGML/tree/main
	//opts.Model = "/Users/me/models/7B/baize-v2-7b.ggmlv3.q4_0.bin" // NO WAY! Dialog generation only and crazy loops

	// https://huggingface.co/TheBloke/Samantha-7B-GGML/tree/main
	//opts.Model = "/Users/me/models/7B/Samantha-7B.ggmlv3.q4_0.bin"

	// ==== 13B ====

	// https://huggingface.co/eachadea/ggml-vicuna-13b-1.1
	//opts.Model = "/Users/me/models/13B/ggml-vic13b-q4_0.bin"

	// https://huggingface.co/execveat/wizardLM-13b-ggml-4bit/tree/main
	//opts.Model = "/Users/me/models/13B/wizardml-13b-q4_0.bin" // usable but no mirostat
	//opts.Model = "/Users/me/models/13B/WizardML-Unc-13b-Q5_1.bin" // släktet - so so

	// https://huggingface.co/TheBloke/wizard-vicuna-13B-GGML/tree/main
	//opts.Model = "/Users/me/models/13B/wizard-vicuna-13B.ggml.q4_0.bin" // no way with topK, try more with mirostat

	// https://huggingface.co/TheBloke/Wizard-Vicuna-13B-Uncensored-GGML/tree/main
	//opts.Model = "/Users/me/models/13B/Wizard-Vicuna-13B-Uncensored.ggml.q4_0.bin" // v2
	//opts.Model = "/Users/me/models/13B/Wizard-Vicuna-13B-Uncensored.ggml.q8_0.bin"

	// -- v2

	// https://huggingface.co/TheBloke/gpt4-x-vicuna-13B-GGML/tree/main
	//opts.Model = "/Users/me/models/13B/gpt4-x-vicuna-13B.ggml.q4_0.bin" // good, great with ### Instruction: { prompt } ### Response:

	// https://huggingface.co/TheBloke/Wizard-Vicuna-13B-Uncensored-GGML/tree/main
	//opts.Model = "/Users/me/models/13B/Wizard-Vicuna-13B-Uncensored.ggml.q4_0.bin" // goood with mirostat! worse with topK! and use ###  prompt
	//opts.Model = "/Users/me/models/13B/Wizard-Vicuna-13B-Uncensored.ggml.q5_1.bin" // good with mirostat and topK too! use ###

	// https://www.reddit.com/r/LocalLLaMA/comments/13igxvs/new_unfiltered_13b_openaccess_ai_collectives/
	// https://huggingface.co/TheBloke/wizard-mega-13B-GGML/resolve/main/wizard-mega-13B.ggml.q4_0.bin
	//opts.Model = "/Users/me/models/13B/wizard-mega-13B.ggml.q4_0.bin" // gooood! with Russian too | instruct and chat
	//opts.Model = "/Users/me/models/13B/wizard-mega-13B.ggml.q5_1.bin"

	// https://huggingface.co/TheBloke/WizardLM-13B-Uncensored-GGML/tree/main
	//opts.Model = "/Users/me/models/13B/wizardLM-13B-Uncensored.ggml.q4_0.bin" // släktet | 100% instruct model | not so good with Russian?

	// https://huggingface.co/TheBloke/Manticore-13B-GGML/tree/main
	//opts.Model = "/Users/me/models/13B/Manticore-13B.ggmlv2.q4_0.bin" // use [ ### Instruction: ... ### Response: ] format without newlines
	//opts.Model = "/Users/me/models/13B/Manticore-13B.ggmlv2.q5_1.bin" // use [ ### Instruction: ... ### Assistant: ] without newlines will work too
	//opts.Model = "/Users/me/models/13B/Manticore-13B.ggmlv2.q8_0.bin"

	// -- v3

	//opts.Model = "/Users/me/models/13B/llama-13b-ggml-v3-q4_0.bin"

	// https://huggingface.co/TheBloke/wizard-mega-13B-GGML/tree/main
	//opts.Model = "/Users/me/models/13B/wizard-mega-13B.ggmlv3.q4_0.bin"
	//opts.Model = "/Users/me/models/13B/wizard-mega-13B.ggmlv3.q5_1.bin"

	// https://huggingface.co/TheBloke/Manticore-13B-GGML/tree/main
	//opts.Model = "/Users/me/models/13B/Manticore-13B.ggmlv3.q4_0.bin" // Very Good! See comparison GDocs
	//opts.Model = "/Users/me/models/13B/Manticore-13B.ggmlv3.q5_0.bin"

	// https://huggingface.co/jondurbin/airoboros-13b
	// https://huggingface.co/latimar/airoboros-13b-ggml/tree/main
	//opts.Model = "/Users/me/models/13B/airoboros-13B.q4_0.bin"
	//opts.Model = "/Users/me/models/13B/airoboros-13B.q4_1.bin"

	// https://huggingface.co/latimar/airoboros-13b-ggml/tree/main
	//opts.Model = "/Users/me/models/13B/airoboros-13B.q5_1.bin"

	// https://huggingface.co/TheBloke/manticore-13b-chat-pyg-GGML/tree/main
	//opts.Model = "/Users/me/models/13B/Manticore-13B-Chat-Pyg.ggmlv3.q4_0.bin"
	//opts.Model = "/Users/me/models/13B/Manticore-13B-Chat-Pyg.ggmlv3.q4_1.bin"
	//opts.Model = "/Users/me/models/13B/Manticore-13B-Chat-Pyg.ggmlv3.q5_0.bin"
	//opts.Model = "/Users/me/models/13B/Manticore-13B-Chat-Pyg.ggmlv3.q5_1.bin"
	//opts.Model = "/Users/me/models/13B/Manticore-13B-Chat-Pyg.ggmlv3.q8_0.bin"

	// https://huggingface.co/TheBloke/guanaco-13B-GGML/tree/main
	//opts.Model = "/Users/me/models/13B/guanaco-13B.ggmlv3.q4_0.bin"
	//opts.Model = "/Users/me/models/13B/guanaco-13B.ggmlv3.q5_0.bin"

	// https://huggingface.co/TheBloke/wizardLM-13B-1.0-GGML/tree/main
	//opts.Model = "/Users/me/models/13B/WizardLM-13B-1.0.ggmlv3.q4_0.bin"
	//opts.Model = "/Users/me/models/13B/WizardLM-13B-1.0.ggmlv3.q4_1.bin"
	//opts.Model = "/Users/me/models/13B/WizardLM-13B-1.0.ggmlv3.q5_0.bin"
	//opts.Model = "/Users/me/models/13B/WizardLM-13B-1.0.ggmlv3.q5_1.bin"
	//opts.Model = "/Users/me/models/13B/WizardLM-13B-1.0.ggmlv3.q8_0.bin"

	// https://huggingface.co/TheBloke/tulu-13B-GGML
	//opts.Model = "/Users/me/models/13B/tulu-13b.ggmlv3.q4_0.bin"
	//opts.Model = "/Users/me/models/13B/tulu-13b.ggmlv3.q4_K_S.bin"

	// ==== 30B ====

	// https://huggingface.co/MetaIX/GPT4-X-Alpaca-30B-4bit/tree/main
	//opts.Model = "/Users/me/models/30B/gpt4-x-alpaca-30b-ggml-q4_1.bin" // maybe with mirostat ?

	// -- v2

	// https://huggingface.co/TheBloke/OpenAssistant-SFT-7-Llama-30B-GGML/tree/main
	//opts.Model = "/Users/me/models/30B/OpenAssistant-SFT-7-Llama-30B.ggml.q4_0.bin" // broken? not, bad bad

	// -- v3

	//opts.Model = "/Users/me/models/30B/WizardLM-30B-Uncensored.ggmlv3.q4_0.bin"
	//opts.Model = "/Users/me/models/30B/WizardLM-30B-Uncensored.ggmlv3.q4_0.bin"

	// https://huggingface.co/TheBloke/Wizard-Vicuna-30B-Uncensored-GGML/tree/main
	//opts.Model = "/Users/me/models/30B/Wizard-Vicuna-30B-Uncensored.ggmlv3.q4_0.bin"

	//opts.Model = "/Users/me/models/30B/guanaco-33B.ggmlv3.q4_0.bin" // Too wild for instruct mode

	// https://huggingface.co/TheBloke/samantha-33B-GGML/tree/main
	//opts.Model = "/Users/me/models/30B/samantha-33B.ggmlv3.q4_0.bin"

	// https://huggingface.co/TheBloke/hippogriff-30b-chat-GGML/tree/main
	//opts.Model = "/Users/me/models/30B/hippogriff-30b.ggmlv3.q4_0.bin"

	// if config was read from file and thus has meaningful settings, go init from there. otherwise use CLI settings
	if conf.ID != "" {
		server.InitFromConfig(&conf, log)
	} else {
		server.Init(
			opts.Host, opts.Port,
			log,
			opts.Pods, opts.Threads, opts.GPULayers,
			opts.Model,
			opts.Prefix, opts.Suffix,
			int(opts.Context), int(opts.Predict),
			opts.Mirostat, opts.MirostatTAU, opts.MirostatETA,
			opts.Temp, opts.TopK, opts.TopP,
			opts.RepeatPenalty, opts.RepeatLastN,
			opts.Deadline,
			opts.Seed)
	}

	// --- Debug output of results and stop after 1 hour in case of running withous --server flag

	if opts.Debug {
		go func() {
			//iter := 0
			for {

				Colorize("\n[magenta]============== JOBS ==============\n")

				for _, job := range server.Jobs {

					var output string
					//if job.Status == "finished" {
					//	output = server.Jobs[job.ID].Output
					//} else {
					output = C.GoString(C.status(C.CString(job.ID)))
					//}

					Colorize("\n[light_magenta]%s [ %s ] | [yellow]%s | [ %d ] tokens | [ %d ] ms. per token [light_blue]| %s\n",
						job.ID,
						job.Model,
						job.Status,
						job.TokenCount,
						job.TokenEval,
						output)
				}

				if server.GoShutdown && len(server.Queue) == 0 && server.RunningThreads == 0 {
					break
				}

				time.Sleep(2 * time.Second)
				//iter++
				//if iter > 600 {
				//	Colorize("\n[light_magenta][STOP][yellow] Time limit 600 * 3 seconds is over!")
				//	break
				//}
			}
		}()
	}

	if !opts.Server {
		Colorize("\n[light_magenta][ INIT ][light_blue] REST API running on [light_magenta]%s:%s", opts.Host, opts.Port)
	}
	log.Infof("[START] REST API running on %s:%s", opts.Host, opts.Port)

	server.Run()
}

func parseOptions() *Options {

	var opts Options

	_, err := flags.Parse(&opts)
	if err != nil {
		Colorize("\n[magenta][ ERROR ][white] Can't parse options from command line! %s\n\n", err.Error())
		os.Exit(0)
	}

	if opts.Server == false && opts.Model == "" {
		Colorize("\n[magenta][ ERROR ][white] Please specify correct model path with [light_magenta]--model[white] parameter!\n\n")
		os.Exit(0)
	}

	if opts.Server == false && opts.Prompt == "" && len(os.Args) > 1 && os.Args[1] != "load" {
		Colorize("\n[magenta][ ERROR ][white] Please specify correct prompt with [light_magenta]--prompt[white] parameter!\n\n")
		os.Exit(0)
	}

	if opts.Pods == 0 {
		opts.Pods = 1
	}

	// Allow to use ALL cores for the program itself and CLI specified number of cores for the parallel tensor math
	// TODO Optimize default settings for CPUs with P and E cores like M1 Pro = 8 performant and 2 energy cores

	if opts.Threads == 0 {
		opts.Threads = int64(runtime.NumCPU())
	}

	if opts.Host == "" {
		opts.Host = "localhost"
	}

	if opts.Port == "" {
		opts.Port = "8080"
	}

	if opts.Context == 0 {
		opts.Context = 1024
	}

	if opts.Predict == 0 {
		opts.Predict = 512
	}

	if opts.Mirostat == 0 {
		opts.Mirostat = 0
	}

	if opts.MirostatTAU == 0 {
		opts.MirostatTAU = 0.2
	}

	if opts.MirostatETA == 0 {
		opts.MirostatETA = 0.1
	}

	if opts.Temp == 0 {
		opts.Temp = 0.2
	}

	if opts.TopK == 0 {
		opts.TopK = 10
	}

	if opts.TopP == 0 {
		opts.TopP = 0.6
	}

	if opts.RepeatPenalty == 0 {
		opts.RepeatPenalty = 1.1
	}

	if opts.RepeatLastN == 0 {
		opts.RepeatLastN = -1
	}

	if opts.Server && !opts.Debug {
		doPrint = false
	}

	if opts.Server {
		doLog = true
	}

	return &opts
}

// Colorize is a wrapper for colorstring.Color() and fmt.Fprintf()
// Join colorstring and go-colorable to allow colors both on Mac and Windows
// TODO: Implement as a small library
func Colorize(format string, opts ...interface{}) (n int, err error) {
	if !doPrint {
		return
	}
	var DefaultOutput = colorable.NewColorableStdout()
	return fmt.Fprintf(DefaultOutput, colorstring.Color(format), opts...)
}

func showLogo() {

	// Rozzo + 3-D + some free time
	// https://patorjk.com/software/taag/#p=display&f=3-D&t=llama.go%0A%0ALLaMA.go
	// Isometric 1, Modular, Rectangles, Rozzo, Small Isometric 1, 3-D

	logo := `                                                    
  /88       /88         /888/888   /88/8888/88   /888/888  /888/8888 /888/888   /888/888    
  /888      /888      /888/ /888 /888/8888/888 /888/ /888  ///8888/ /8888//888 /8888//888  
  /8888/88  /8888/88  /8888/8888 /888/8888/888 /8888/8888  /8888/   /888 /8888 /888 /8888 
  /8888/888 /8888/888 /888 /8888 /888//88 /888 /888 /8888 /8888/888 //888/888  //888/888
  //// ///  //// ///  ///  ////  ///  //  ///  ///  ////  //// ///   /// ///    /// ///`

	logoColored := ""
	prevColor := ""
	color := ""
	line := 0
	colors := []string{"[black]", "[light_blue]", "[magenta]", "[light_magenta]", "[light_blue]"}

	for _, char := range logo {
		if char == '\n' {
			line++
		} else if char == '/' {
			color = "[blue]"
		} else if char == '8' {
			color = colors[line]
			char = '▒'
		}
		if color == prevColor {
			logoColored += string(char)
		} else {
			logoColored += color + string(char)
		}
	}

	Colorize(logoColored)
	Colorize(
		"\n\n   [magenta]▒▒▒▒▒[light_magenta] [ LLaMAZoo v" +
			VERSION +
			" ] [light_blue][ Platform for serving any GPT model of LLaMA family ] [magenta]▒▒▒▒▒\n\n")
}
