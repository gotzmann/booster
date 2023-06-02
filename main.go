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
// void loop(void * ctx, char * jobID, char * prompt);
// const char * status(char * jobID);

// #cgo LDFLAGS: bridge.o ggml.o llama.o -lstdc++ -framework Accelerate

/*
#include <stdint.h>
#include <bridge.h>
//const char * status(char * jobID);
//int64_t timing(char * jobID);
#cgo CFLAGS:   -I. -O3 -DNDEBUG -fPIC -pthread -std=c17
#cgo CXXFLAGS: -I. -O3 -DNDEBUG -fPIC -pthread -std=c++17
#cgo LDFLAGS: bridge.o ggml.o llama.o -lstdc++
*/
import "C"

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"time"

	config "github.com/golobby/config/v3"
	"github.com/golobby/config/v3/pkg/feeder"
	flags "github.com/jessevdk/go-flags"
	colorable "github.com/mattn/go-colorable"
	"github.com/mitchellh/colorstring"
	"github.com/pkg/profile"

	"github.com/gotzmann/llamazoo/pkg/server"
)

const VERSION = "0.9.8"

type Options struct {
	Prompt  string  `long:"prompt" description:"Text prompt from user to feed the model input"`
	Model   string  `long:"model" description:"Path and file name of converted .bin LLaMA model [ llama-7b-fp32.bin, etc ]"`
	Prefix  string  `long:"prefix" description:"Prompt prefix if needed, like \"### Instruction:\""`
	Suffix  string  `long:"suffix" description:"Prompt suffix if needed, like \"### Response:\""`
	Seed    uint32  `long:"seed" description:"Seed number for random generator initialization [ current Unix time by default ]"`
	Server  bool    `long:"server" description:"Start in Server Mode acting as REST API endpoint"`
	Host    string  `long:"host" description:"Host to allow requests from in Server Mode [ localhost by default ]"`
	Port    string  `long:"port" description:"Port listen to in Server Mode [ 8080 by default ]"`
	Pods    int     `long:"pods" description:"Maximum pods or units of parallel execution allowed in Server Mode [ 1 by default ]"`
	Threads int64   `long:"threads" description:"Max number of CPU cores you allow to use for one pod [ all cores by default ]"`
	Context uint32  `long:"context" description:"Context size in tokens [ 1024 by default ]"`
	Predict uint32  `long:"predict" description:"Number of tokens to predict [ 512 by default ]"`
	Temp    float32 `long:"temp" description:"Model temperature hyper parameter [ 0.50 by default ]"`
	Silent  bool    `long:"silent" description:"Hide welcome logo and other output [ shown by default ]"`
	Chat    bool    `long:"chat" description:"Chat with user in interactive mode instead of compute over static prompt"`
	Dir     string  `long:"dir" description:"Directory used to download .bin model specified with --model parameter [ current by default ]"`
	Profile bool    `long:"profile" description:"Profe CPU performance while running and store results to cpu.pprof file"`
	UseAVX  bool    `long:"avx" description:"Enable x64 AVX2 optimizations for Intel and AMD machines"`
	UseNEON bool    `long:"neon" description:"Enable ARM NEON optimizations for Apple and ARM machines"`
	Ignore  bool    `long:"ignore" description:"Ignore server JSON and YAML configs, use only CLI params"`
}

func main() {

	// Last resort in case of panic while running
	//defer func() {
	//	reason := recover()
	//	if reason != nil {
	//		Colorize("\n[magenta][ ERROR ][white] %s\n\n", reason)
	//		os.Exit(0)
	//	}
	//}()

	// TODO: Allow to overwrite some options from the command-line
	opts := parseOptions()

	if opts.Profile {
		defer profile.Start(profile.ProfilePath(".")).Stop()
	}

	if !opts.Silent {
		showLogo()
	}

	// --- read config from JSON or YAML

	conf := server.Config{}
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
	/*
		// --- set model parameters from user settings and safe defaults

		server.Params = &llama.ModelParams{
			Model: opts.Model,

			MaxThreads: opts.Threads,

			UseAVX:  opts.UseAVX,
			UseNEON: opts.UseNEON,

			Interactive: opts.Chat,

			CtxSize:      opts.Context,
			Seed:         -1,
			PredictCount: opts.Predict,
			RepeatLastN:  opts.Context, // TODO: Research on best value
			PartsCount:   -1,
			BatchSize:    opts.Context, // TODO: What's the better size?

			TopK:          40,
			TopP:          0.95,
			Temp:          opts.Temp,
			RepeatPenalty: 1.10,

			MemoryFP16: true,
		} */

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
	opts.Model = "/Users/me/models/13B/Manticore-13B-Chat-Pyg.ggmlv3.q4_0.bin"
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

	/*

		TODO: If there no file on path, CGO panics

			libc++abi: terminating due to uncaught exception of type std::runtime_error: failed to open /Users/me/models/30B/llama-ggml-v2-q4_0.bin: No such file or directory
		SIGABRT: abort
		PC=0x19b340724 m=0 sigcode=0
		signal arrived during cgo execution

		goroutine 1 [syscall]:
		runtime.cgocall(0x104d50d50, 0x14000147d58)
			/opt/homebrew/Cellar/go/1.20.4/libexec/src/runtime/cgocall.go:157 +0x54 fp=0x14000147d20 sp=0x14000147ce0 pc=0x104935d74
		github.com/gotzmann/llamazoo/pkg/server._Cfunc_initFromParams(0x6000031b0000, 0x6, 0x400, 0x100, 0x3f4ccccd, 0x6458fe8c)
			_cgo_gotypes.go:89 +0x38 fp=0x14000147d50 sp=0x14000147d20 pc=0x104d4b688
		github.com/gotzmann/llamazoo/pkg/server.Init({0x104d96b11?, 0x10526c2c0?}, {0x104d94af8?, 0x1051b8f08?}, 0x1, 0x6, {0x104daa382, 0x2b}, 0x400, 0x100, ...)
			/Users/me/git/llamazoo/pkg/server/server.go:163 +0x180 fp=0x14000147df0 sp=0x14000147d50 pc=0x104d4bbf0
		main.main()
			/Users/me/git/llamazoo/main.go:189 +0x304 fp=0x14000147f70 sp=0x14000147df0 pc=0x104d4fa54
		runtime.main()
			/opt/homebrew/Cellar/go/1.20.4/libexec/src/runtime/proc.go:250 +0x248 fp=0x14000147fd0 sp=0x14000147f70 pc=0x10496a2c8
		runtime.goexit()
			/opt/homebrew/Cellar/go/1.20.4/libexec/src/runtime/asm_arm64.s:1172 +0x4 fp=0x14000147fd0 sp=0x14000147fd0 pc=0x10499cea4

	*/

	// if config was read from file and thus has meaningful settings, go init from there. otherwise use CLI settings
	if conf.ID != "" {
		server.InitFromConfig(&conf)
	} else {
		server.Init(
			opts.Host,
			opts.Port,
			opts.Pods,
			opts.Threads,
			opts.Model,
			opts.Prefix,
			opts.Suffix,
			int(opts.Context),
			int(opts.Predict),
			opts.Temp,
			opts.Seed)
	}

	// --- wait for API calls as REST server, or compute just the one prompt from user CLI

	// TODO: Control signals between main() and server
	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		iter := 0
		for {

			fmt.Printf("\n")

			//Colorize("\n[magenta]============== queue ==============")
			//for job := range server.Queue {
			//	Colorize("\n[light_magenta]%s", job)
			//}

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

			time.Sleep(5 * time.Second)
			iter++
			if iter > 600 {
				Colorize("\n[light_magenta][STOP][yellow] Time limit 600 * 5 seconds is over!")
				break
			}

		}
		wg.Done()
	}()

	go server.Run()

	if !opts.Silent && opts.Server {
		Colorize("\n[light_magenta][ INIT ][light_blue] REST server ready on [light_magenta]%s:%s", opts.Host, opts.Port)
	}

	wg.Wait()
}

func parseOptions() *Options {

	var opts Options

	_, err := flags.Parse(&opts)
	if err != nil {
		Colorize("\n[magenta][ ERROR ][white] Can't parse options from command line!\n\n")
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

	if opts.Temp == 0 {
		opts.Temp = 0.80
	}

	return &opts
}

// Colorize is a wrapper for colorstring.Color() and fmt.Fprintf()
// Join colorstring and go-colorable to allow colors both on Mac and Windows
// TODO: Implement as a small library
func Colorize(format string, opts ...interface{}) (n int, err error) {
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
