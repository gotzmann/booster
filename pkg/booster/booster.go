package booster

// TODO: Allow short locales: en -> en_US, ru -> ru_RU
// TODO: Experiment with the batch size
// TODO: If there no batch size in config - server do not work
// TODO: Loading model... before REST API running
// TODO: Watchdog exceptions: CUDA error 2 at ggml-cuda.cu:7233: out of memory
// TODO: Init Janus Sampling from CLI
// TODO: Update code for maintain session files for GGUF format (tokenization BOS, etc)
// TODO: Protect user input from injection of PROMPT attacs, like USER: or ASSISTANT: wording
// TODO: Use UUID instead of string https://github.com/google/uuid/blob/master/uuid.go
// TODO: Benchmark map[string] vs map[UUID] by memory and performance for accessing 1 million elements
// wiki-raw datasets https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/

/*
#include <stdlib.h>
#include <stdint.h>
const char * status(char * jobID);
uint32_t getSeed(char * jobID);
int64_t getPromptTokenCount(char * jobID);
*/
import "C"

import (
	"bufio"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"syscall"
	"time"

	config "github.com/golobby/config/v3"
	"github.com/golobby/config/v3/pkg/feeder"
	"github.com/google/uuid"
	flags "github.com/jessevdk/go-flags"
	colorable "github.com/mattn/go-colorable"
	"github.com/mitchellh/colorstring"
	"github.com/pkg/profile"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/gotzmann/booster/pkg/server"
)

const VERSION = "3.0.0"

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
	BatchSize     int64   `long:"batch-size" description:"Batch size (in tokens) for one GPU inference flow [ 512 by default ]"`
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
	Swap          string  `long:"swap" description:"Path for user session swap files [ only for CPU inference, up to 1Gb per each ]"`
	MaxSessions   int     `long:"max-sessions" description:"How many sessions allowed to be stored on disk [ unlimited by default ]"`
}

var (
	doPrint bool = true
	doLog   bool = false
	conf    server.Config
)

func Run() {

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

		// -- user-friendly naming for some fields

		for _, sampling := range conf.Samplings {

			if sampling.Temperature == 0.0 && sampling.Temp != 0.0 {
				sampling.Temperature = sampling.Temp
			}

			if sampling.TopK == 0 && sampling.Top_K != 0 {
				sampling.TopK = sampling.Top_K
			}

			if sampling.TopP == 0.0 && sampling.Top_P != 0.0 {
				sampling.TopP = sampling.Top_P
			}

			if sampling.RepetitionPenalty == 0.0 && sampling.Repetition_Penalty != 0.0 {
				sampling.RepetitionPenalty = sampling.Repetition_Penalty
			}
		}

		// fmt.Printf("%+v", conf) // DEBUG
	}

	if opts.Profile {
		defer profile.Start(profile.ProfilePath(".")).Stop()
	}

	var zapWriter zapcore.WriteSyncer
	zapConfig := zap.NewProductionEncoderConfig()
	zapConfig.NameKey = "booster" // TODO: pod name from config?
	//zapConfig.CallerKey = ""       // do not log caller like "booster/booster.go:156"
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

	// -- Show logo and some useful info
	//    TODO: More grained info [ GPU, RAM, etc ]

	model := "undefined"
	sampling := "default"
	if conf.ID != "" {
		// model = conf.Models[0].Path
		model = reflect.ValueOf(conf.Models).MapKeys()[0].String()
		log.Infof("[ DEBUG ] MODEL = %s", model)
		// if conf.Models[0].Janus > 0 {
		sampling = reflect.ValueOf(conf.Samplings).MapKeys()[0].String()
		log.Infof("[ DEBUG ] SAMPLING = %s", sampling)
		//if conf.Models[0].Janus > 0 {
		//	sampling = "Janus v1"
		//}
	} else {
		model = opts.Model
		sampling = "default"
	}
	if !opts.Server || opts.Debug {
		showLogo(conf.Models[model].Path, sampling)
	}

	log.Infof("[ START ] Booster v%s is starting...", VERSION)

	// --- Allow graceful shutdown via OS signals
	// https://ieftimov.com/posts/four-steps-daemonize-your-golang-programs/

	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM)

	// --- Listen for OS signals in background when Server mode

	go func() {

		<-signalChan

		// -- break execution immediate when DEBUG

		if opts.Debug || !opts.Server {
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
		Colorize("\n[light_magenta][ STOP ][light_blue] Booster was stopped. Arrivederci!\n\n")
		log.Info("[ STOP ] Booster was stopped. Arrivederci!")
		logger.Sync()
	}()

	// if config was read from file and thus has meaningful settings, go init from there. otherwise use CLI settings
	if conf.ID != "" {
		server.InitFromConfig(&conf, log)
	} else {
		server.Init(
			opts.Host, opts.Port,
			log,
			1, opts.Threads,
			0, 0, 0, 0,
			opts.Model,
			opts.Preamble, opts.Prefix, opts.Suffix,
			int(opts.Context), int(opts.Predict),
			opts.Mirostat, opts.MirostatENT, opts.MirostatLR,
			opts.Temp, opts.TopK, opts.TopP,
			opts.TypicalP,
			opts.PenaltyRepeat, opts.PenaltyLastN,
			opts.Deadline,
			opts.Seed,
			opts.Swap,
			"",
		)
	}

	// --- Interactive mode for chatting with models from command line

	if !opts.Server {
		go func() {

			time.Sleep(1 * time.Second)
			sessionID := uuid.New().String()
			Colorize("\n[magenta][ INIT ][light_blue] Interactive mode, enter your prompts and commands below")

			for {

				Colorize("\n\n[magenta]>>> [light_magenta]")
				prompt, _ := bufio.NewReader(os.Stdin).ReadString('\n')

				jobID := uuid.New().String()
				server.PlaceJob(jobID, "" /* payload.Model */, sessionID, prompt)

				prevOutput := ""
				history := server.Sessions[sessionID]

				Colorize("\n[blue]<<< [light_blue]")

				for {

					time.Sleep(1 * time.Second)
					server.Mutex.Lock()

					// history was shrunk after job really started and detected that context is too long
					if server.Jobs[jobID].Status != "finished" && history != server.Sessions[sessionID] {
						history = server.Sessions[sessionID]
					}

					output := C.GoString(C.status(C.CString(jobID)))
					// waiting while prompt history will be processed completely
					if server.Jobs[jobID].Status == "processing" && len(output) < len(server.Jobs[jobID].FullPrompt) {
						server.Mutex.Unlock()
						continue
					}
					output, _ = strings.CutPrefix(output, server.Jobs[jobID].FullPrompt)

					if server.Jobs[jobID].Status == "finished" {
						assistantTemplate := server.Prompts[server.Jobs[jobID].PromptID].Templates.Assistant
						if strings.Contains(assistantTemplate, "{ASSISTANT}") {
							cut := strings.Index(assistantTemplate, "{ASSISTANT}") + len("{ASSISTANT}")
							assistantSuffix := assistantTemplate[cut:]
							output, _ = strings.CutSuffix(output, assistantSuffix)
						}
					}

					if len(output) > len(prevOutput) {
						Colorize("[light_blue]%s", output[len(prevOutput):])
						prevOutput = output
					}

					if server.Jobs[jobID].Status == "finished" {
						server.Mutex.Unlock()
						break
					}

					server.Mutex.Unlock()
				}
			}

		}()
	}

	// --- Debug output of results and stop after 1 hour in case of running withous --server flag

	if opts.Debug {
		go func() {

			jobCount := 0
			needUpdate := false

			for {

				time.Sleep(5 * time.Second)

				for _, job := range server.Jobs {
					if job.Status == "processing" || job.Status == "queued" {
						needUpdate = true
						break
					}
				}

				if !needUpdate && jobCount == len(server.Jobs) {
					continue
				}

				Colorize("\n\n[magenta]======================================================== [ JOBS ] ========================================================\n")

				// TODO: Show jobs in timing order (need extra slice)
				for _, job := range server.Jobs {

					output := C.GoString(C.status(C.CString(job.ID)))
					// FIXME: Avoid LLaMA v2 leading space
					//if len(output) > 0 && output[0] == ' ' {
					//	output = output[1:]
					//}

					podID := "--"
					if job.Pod != nil {
						podID = job.Pod.ID
					}

					Colorize("\n[light_magenta]%s [light_green][ %s ] [light_yellow][ %s ] [light_magenta][ %s ] [light_gray]TOKENS: IN [ %d => %d ] OUT || MILLISECONDS: IN [ %d => %d ] OUT || SEED: %d [light_blue]\n\n%s\n",
						job.ID,
						job.Status,
						podID,
						job.ModelID,
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

				if !needUpdate {
					jobCount = len(server.Jobs)
				}

				needUpdate = false
			}
		}()
	}

	if opts.Server {
		server.Run(!opts.Server || opts.Debug)
	} else {
		go server.Engine(nil)
		<-signalChan
	}
}

func parseOptions() *Options {

	var opts Options

	_, err := flags.Parse(&opts)
	if err != nil {
		Colorize("\n[magenta][ ERROR ][white] Can't parse options from command line! %s\n\n", err.Error())
		os.Exit(0)
	}

	/*
		FIXME: Do we need these checks when there INTERACTIVE mode by default?

		if opts.Server == false && opts.Model == "" {
			Colorize("\n[magenta][ ERROR ][white] Please specify correct model path with [light_magenta]--model[white] parameter!\n\n")
			os.Exit(0)
		}

		if opts.Server == false && opts.Prompt == "" && len(os.Args) > 1 && os.Args[1] != "load" {
			Colorize("\n[magenta][ ERROR ][white] Please specify correct prompt with [light_magenta]--prompt[white] parameter!\n\n")
			os.Exit(0)
		}
	*/

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

func showLogo(model, sampling string) {

	model = filepath.Base(model)
	model = strings.TrimSuffix(model, filepath.Ext(model))

	// Rozzo + 3-D + some free time
	// https://patorjk.com/software/taag/#p=display&f=3-D&t=llama.go%0A%0ALLaMA.go
	// Some other interesting options: Isometric 1, Modular, Rectangles, Rozzo, Small Isometric 1, 3-D

	logo := `
  /88888/888  /88888/8888 /88888/8888 /8888/8888 /88888/8888 /8888/8888  /8888/888
  /8888/88888 /8888 //888 /8888 //888 /88888///  /8888/88888 /888/88//   /888 //8888
  /88888//888 /888  /8888 /888  /8888 ////888888 ///888/88/  /8888/888   /8888/8888
  /8888/88888 /8888/88888 /8888/88888 /8888/8888   /888/88   /8888/88888 /8888//8888
  ///// ////  //// /////  //// /////  //// ////    /// //    //// /////  ////  ////`

	/*
	   	logo := `
	     /8888/888   /8888/888  /88       /88       /8888 /8888/888   /8888/888  /8888/888
	    /8888 ///88 /8888 //888 /888      /888      /8888 /8888//8888 /8888///   /8888///888
	    /888    //  /888  /8888 /8888/88  /8888/88  /8888 /8888 /8888 /8888/888  /8888/8888
	    //8888/888  //888/8888  /8888/888 /8888/888 /8888 /8888/8888  /8888/8888 /8888//8888
	     //// ///    /// ////   //// ///  //// ///  ////  //// ////   //// ////  ////  ////`
	*/
	/*
	   	logo := `
	     /88       /88         /888/888   /88/8888/88   /888/888  /888/8888 /888/888   /888/888
	     /888      /888      /888/ /888 /888/8888/888 /888/ /888  ///8888/ /8888//888 /8888//888
	     /8888/88  /8888/88  /8888/8888 /888/8888/888 /8888/8888  /8888/   /888 /8888 /888 /8888
	     /8888/888 /8888/888 /888 /8888 /888//88 /888 /888 /8888 /8888/888 //888/888  //888/888
	     //// ///  //// ///  ///  ////  ///  //  ///  ///  ////  //// ///   /// ///    /// ///`
	*/

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

	Colorize("\n\n  [magenta]===[light_magenta] [ Booster v" + VERSION +
		" ] [light_blue][ The Open Platform for serving Large Language Models ] [magenta]===\n")

	Colorize("\n  [magenta][    model ][light_magenta] " + model)
	Colorize("\n  [blue][ sampling ][light_blue] " + sampling + "\n")
}
