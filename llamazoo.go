package main

// TODO: Watchdog exceptions: CUDA error 2 at ggml-cuda.cu:7233: out of memory
// TODO: Init Janus Sampling from CLI
// TODO: Update code for maintain session files for GGUF format (tokenization BOS, etc)
// TODO: Protect user input from injection of PROMPT attacs, like USER: or ASSISTANT: wording
// TODO: Use UUID instead of string https://github.com/google/uuid/blob/master/uuid.go
// TODO: Benchmark map[string] vs map[UUID] by memory and performance for accessing 1 million elements
// wiki-raw datasets https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/

// ggml-cuda.o

/*
#include <stdlib.h>
#include <stdint.h>
const char * status(char * jobID);
uint32_t getSeed(char * jobID);
int64_t getPromptTokenCount(char * jobID);
#cgo linux CFLAGS:   -std=c17   -O3 -I.           -fPIC -pthread -march=native -mtune=native -DNDEBUG -DGGML_USE_K_QUANTS -DGGML_USE_CUBLAS -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DLOG_DISABLE_LOGS -I/usr/local/cuda/include -I/opt/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include
#cgo linux CXXFLAGS: -std=c++17 -O3 -I. -Icommon  -fPIC -pthread -march=native -mtune=native -DNDEBUG -DGGML_USE_K_QUANTS -DGGML_USE_CUBLAS -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DLOG_DISABLE_LOGS -I/usr/local/cuda/include -I/opt/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include
#cgo linux LDFLAGS: bridge.o janus.o ggml.o ggml-backend.o ggml-alloc.o llama.o k_quants.o -lstdc++ -lm -lcublas -lculibos -lcudart -lcublasLt -lpthread -ldl -lrt -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -L/usr/local/cuda/targets/x86_64-linux/lib
#cgo darwin CFLAGS:   -I. -O3 -fPIC -pthread -std=c17 -DNDEBUG -DGGML_USE_METAL -DGGML_METAL_NDEBUG -mcpu=native
#cgo darwin CXXFLAGS: -I. -Icommon -O3 -fPIC -pthread -std=c++17 -DNDEBUG -DGGML_USE_METAL -mcpu=native
#cgo darwin LDFLAGS: bridge.o janus.o ggml.o ggml-backend.o ggml-alloc.o llama.o k_quants.o ggml-metal.o -lstdc++ -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
*/
import "C"

import (
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"strings"
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

const VERSION = "0.43.0"

type Options struct {
	Prompt        string  `long:"prompt" description:"Text prompt from user to feed the model input"`
	Model         string  `long:"model" description:"Path and file name of converted .bin LLaMA model [ llama-7b-fp32.bin, etc ]"`
	Config        string  `long:"config" description:"Use exact config file path [ config.yaml or config.json in current folder by default ]"`
	Preamble      string  `long:"preamble" description:"Preamble for model prompt, like \"You are a helpful AI assistant\""`
	Prefix        string  `long:"prefix" description:"Prompt prefix if needed, like \"### Instruction:\""`
	Suffix        string  `long:"suffix" description:"Prompt suffix if needed, like \"### Response:\""`
	Seed          uint32  `long:"seed" description:"Seed number for random generator initialization [ current Unix time by default ]"`
	Server        bool    `long:"server" description:"Start in Server Mode acting as REST API endpoint"`
	Debug         bool    `long:"debug" description:"Stream debug info to console while processing requests"`
	Log           string  `long:"log" description:"Log file location to save all events in Server mode"`
	Deadline      int64   `long:"deadline" description:"Time in seconds after which unprocessed jobs will be deleted from the queue"`
	Host          string  `long:"host" description:"Host to allow requests from in Server mode [ localhost by default ]"`
	Port          string  `long:"port" description:"Port listen to in Server Mode [ 8080 by default ]"`
	Threads       int64   `long:"threads" description:"Max number of CPU cores you allow to use for one pod [ all cores by default ]"`
	GPUs          []int64 `long:"gpus" description:"Specify GPU split for each pod when there GPUs (one or more) are available"`
	Context       uint32  `long:"context" description:"Context size in tokens [ 2048 by default ]"`
	Predict       uint32  `long:"predict" description:"Number of tokens to predict [ 1024 by default ]"`
	Janus         uint32  `long:"janus" description:"Janus Sampling version [ not used by default ]"`
	Mirostat      uint32  `long:"mirostat" description:"Mirostat version [ zero or disabled by default ]"`
	MirostatENT   float32 `long:"mirostat-ent" description:"Mirostat target entropy or TAU value [ 0.1 by default ]"`
	MirostatLR    float32 `long:"mirostat-lr" description:"Mirostat Learning Rate or ETA value [ 0.1 by default ]"`
	Temp          float32 `long:"temp" description:"Model temperature hyper parameter [ 0.1 by default ]"`
	TopK          int     `long:"top-k" description:"TopK parameter for the model [ 8 by default ]"`
	TopP          float32 `long:"top-p" description:"TopP parameter for the model [ 0.4 by default ]"`
	TypicalP      float32 `long:"typical-p" description:"TypicalP parameter for the sampling [ 1.0 by default == disabled ]"`
	PenaltyRepeat float32 `long:"penalty-repeat" description:"PenaltyRepeat [ 1.1 by default ]"`
	PenaltyLastN  int     `long:"penalty-last-n" description:"PenaltyLastN [ -1 by default ]"`
	Silent        bool    `long:"silent" description:"Hide welcome logo and other output [ shown by default ]"`
	Chat          bool    `long:"chat" description:"Chat with user in interactive mode instead of compute over static prompt"`
	Dir           string  `long:"dir" description:"Directory used to download .bin model specified with --model parameter [ current by default ]"`
	Profile       bool    `long:"profile" description:"Profe CPU performance while running and store results to cpu.pprof file"`
	UseAVX        bool    `long:"avx" description:"Enable x64 AVX2 optimizations for Intel and AMD machines"`
	UseNEON       bool    `long:"neon" description:"Enable ARM NEON optimizations for Apple and ARM machines"`
	Ignore        bool    `long:"ignore" description:"Ignore server JSON and YAML configs, use only CLI params"`
	Sessions      string  `long:"sessions" description:"Path to where sessions files will be held [ up to 1Gb per each ]"`
	MaxSessions   int     `long:"max-sessions" description:"How many sessions allowed to be stored on disk [ unlimited by default ]"`
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

		if opts.Config != "" {
			if _, err := os.Stat(opts.Config); err != nil {
				Colorize("\n[magenta][ ERROR ][white] Can't find specified config file!\n\n")
				os.Exit(0)
			}
		}

		if opts.Config != "" && strings.Contains(opts.Config, ".json") {
			feed = feeder.Json{Path: opts.Config}
		} else if opts.Config != "" && strings.Contains(opts.Config, ".yaml") {
			feed = feeder.Yaml{Path: opts.Config}
		} else if _, err := os.Stat("config.json"); err == nil {
			feed = feeder.Json{Path: "config.json"}
		} else if _, err := os.Stat("config.yaml"); err == nil {
			feed = feeder.Yaml{Path: "config.yaml"}
		}

		if feed == nil {
			Colorize("\n[magenta][ ERROR ][white] Can't find default config file!\n\n")
			os.Exit(0)
		}

		err := config.New().AddFeeder(feed).AddStruct(&conf).Feed()
		if err != nil {
			Colorize("\n[magenta][ ERROR ][white] Can't parse config file! %s\n\n", err.Error())
			os.Exit(0)
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

	if !opts.Server || opts.Debug {
		showLogo()
	}
	log.Infof("[START] LLaMAZoo v%s starting...", VERSION)

	// --- Allow graceful shutdown via OS signals
	// https://ieftimov.com/posts/four-steps-daemonize-your-golang-programs/

	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM)

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
				pending += 1 /*conf.Pods*/ // TODO: Allow N pods
				Colorize("\n[light_magenta][ STOP ][light_blue] Wait while [light_magenta][ %d ][light_blue] requests will be finished...", pending)
				log.Infof("[STOP] Wait while [ %d ] requests will be finished...", pending)
			}
		}
	}()

	// --- Do all we need in case of graceful shutdown or unexpected panic

	defer func() {
		signal.Stop(signalChan)
		reason := recover()
		if reason != nil {
			Colorize("\n[light_magenta][ ERROR ][white] %s\n\n", reason)
			log.Error("%s", reason)
			os.Exit(0)
		}
		Colorize("\n[light_magenta][ STOP ][light_blue] LLaMAZoo was stopped. Arrivederci!\n\n")
		log.Info("[STOP] LLaMAZoo was stopped. Arrivederci!")
		logger.Sync()
	}()

	// if config was read from file and thus has meaningful settings, go init from there. otherwise use CLI settings
	if conf.ID != "" {
		server.InitFromConfig(&conf, log)
	} else {
		server.Init(
			opts.Host, opts.Port,
			log,
			/*opts.Pods*/ 1, opts.Threads, // TODO: Support N pods
			//opts.GPUs, opts.GPULayers,
			0, 0, // TODO: Support GPUs from command-line
			//NUMA, LowVRAM,
			opts.Model,
			opts.Preamble, opts.Prefix, opts.Suffix,
			int(opts.Context), int(opts.Predict),
			opts.Mirostat, opts.MirostatENT, opts.MirostatLR,
			opts.Temp, opts.TopK, opts.TopP,
			opts.TypicalP,
			opts.PenaltyRepeat, opts.PenaltyLastN,
			opts.Deadline,
			opts.Seed,
			opts.Sessions,
			"", // TODO: Debug Level
		)
	}

	// --- Debug output of results and stop after 1 hour in case of running withous --server flag

	if opts.Debug {
		go func() {
			for {

				Colorize("\n[magenta]======================================== [ JOBS ] ========================================\n")

				for _, job := range server.Jobs {

					var output string
					output = C.GoString(C.status(C.CString(job.ID)))
					// FIXME: Avoid LLaMA v2 leading space
					if len(output) > 0 && output[0] == ' ' {
						output = output[1:]
					}

					Colorize("\n[light_magenta]%s [light_green][ %s ] [dark_gray][ %s ] [dark_gray]%d => %d tokens | %d => %d ms. | seed %d [light_blue]\n\n%s\n",
						job.ID,
						job.Status,
						job.Model,
						C.getPromptTokenCount(C.CString(job.ID)),
						job.OutputTokenCount,
						job.PromptEval,
						job.TokenEval,
						C.getSeed(C.CString(job.ID)),
						output)
				}

				if server.GoShutdown && len(server.Queue) == 0 && server.RunningThreads == 0 {
					break
				}

				time.Sleep(5 * time.Second)
			}
		}()
	}

	server.Run(!opts.Server || opts.Debug)
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
		opts.Context = 2048
	}

	if opts.Predict == 0 {
		opts.Predict = 1024
	}

	if opts.MirostatENT == 0 {
		opts.MirostatENT = 0.1
	}

	if opts.MirostatLR == 0 {
		opts.MirostatLR = 0.1
	}

	if opts.Temp == 0 {
		opts.Temp = 0.1
	}

	if opts.TopK == 0 {
		opts.TopK = 8
	}

	if opts.TopP == 0 {
		opts.TopP = 0.4
	}

	if opts.TypicalP == 0 {
		opts.TypicalP = 1.0 // disabled
	}

	if opts.PenaltyRepeat == 0 {
		opts.PenaltyRepeat = 1.1
	}

	if opts.PenaltyLastN == 0 {
		opts.PenaltyLastN = -1
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
