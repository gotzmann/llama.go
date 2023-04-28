package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/google/uuid"
	flags "github.com/jessevdk/go-flags"
	colorable "github.com/mattn/go-colorable"
	"github.com/mitchellh/colorstring"
	"github.com/pkg/profile"

	"github.com/gotzmann/llama.go/pkg/llama"
	"github.com/gotzmann/llama.go/pkg/server"
)

const VERSION = "1.4.0"

type Options struct {
	Prompt  string  `long:"prompt" description:"Text prompt from user to feed the model input"`
	Model   string  `long:"model" description:"Path and file name of converted .bin LLaMA model [ llama-7b-fp32.bin, etc ]"`
	Server  bool    `long:"server" description:"Start in Server Mode acting as REST API endpoint"`
	Host    string  `long:"host" description:"Host to allow requests from in Server Mode [ localhost by default ]"`
	Port    string  `long:"port" description:"Port listen to in Server Mode [ 8080 by default ]"`
	Pods    int64   `long:"pods" description:"Maximum pods or units of parallel execution allowed in Server Mode [ 1 by default ]"`
	Threads int     `long:"threads" description:"Max number of CPU cores you allow to use for one pod [ all cores by default ]"`
	Context uint32  `long:"context" description:"Context size in tokens [ 1024 by default ]"`
	Predict uint32  `long:"predict" description:"Number of tokens to predict [ 512 by default ]"`
	Temp    float32 `long:"temp" description:"Model temperature hyper parameter [ 0.50 by default ]"`
	Silent  bool    `long:"silent" description:"Hide welcome logo and other output [ shown by default ]"`
	Chat    bool    `long:"chat" description:"Chat with user in interactive mode instead of compute over static prompt"`
	Dir     string  `long:"dir" description:"Directory used to download .bin model specified with --model parameter [ current by default ]"`
	Profile bool    `long:"profile" description:"Profe CPU performance while running and store results to cpu.pprof file"`
	UseAVX  bool    `long:"avx" description:"Enable x64 AVX2 optimizations for Intel and AMD machines"`
	UseNEON bool    `long:"neon" description:"Enable ARM NEON optimizations for Apple and ARM machines"`
}

func main() {

	opts := parseOptions()

	if opts.Profile {
		defer profile.Start(profile.ProfilePath(".")).Stop()
	}

	if !opts.Silent {
		showLogo()
	}

	// --- special command to load model file

	if len(os.Args) > 1 && os.Args[1] == "load" {
		Colorize("[magenta][ LOAD ][light_blue] Downloading model [light_magenta]%s[light_blue] into [light_magenta]%s[light_blue]", opts.Model, opts.Dir)
		size, err := downloadModel(opts.Dir, opts.Model)
		if err != nil {
			Colorize("\n[magenta][ ERROR ][light_blue] Model [light_magenta]%s[light_blue] was not downloaded: [light_red]%s!\n\n", opts.Model, err.Error())
		} else {
			Colorize("\n[magenta][ LOAD ][light_blue] Model [light_magenta]%s[light_blue] of size [light_magenta]%d Gb[light_blue] was successfully downloaded!\n\n", opts.Model, size/1024/1024/1024)
		}
		os.Exit(0)
	}

	// --- set model parameters from user settings and safe defaults

	params := &llama.ModelParams{
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
	}

	// --- load the model and vocab

	vocab, model, err := llama.LoadModel(params.Model, params, opts.Silent)
	if err != nil {
		Colorize("\n[magenta][ ERROR ][white] Failed to load model [light_magenta]\"%s\"\n\n", params.Model)
		os.Exit(0)
	}

	// --- set up internal REST server

	server.MaxPods = opts.Pods
	server.Host = opts.Host
	server.Port = opts.Port
	server.Vocab = vocab
	server.Model = model
	server.Params = params

	go server.Run()

	if !opts.Silent && opts.Server {
		Colorize("\n[light_magenta][ INIT ][light_blue] REST server ready on [light_magenta]%s:%s", opts.Host, opts.Port)
	}

	// --- wait for API calls as REST server, or compute just the one prompt from user CLI

	// TODO: Control signals between main() and server
	var wg sync.WaitGroup
	wg.Add(1)

	if opts.Server {
		wg.Wait()
	} else {

		// add a space to match LLaMA tokenizer behavior
		prompt := " " + opts.Prompt
		jobID := uuid.New().String()
		server.PlaceJob(jobID, prompt)
		output := ""

		//Colorize("\n\n[magenta]▒▒▒[light_yellow]" + prompt + "\n[light_blue]▒▒▒ ")
		Colorize("\n\n[magenta][ PROMPT ][light_magenta]" + prompt + "\n[light_blue][ OUTPUT ][white]")

		for {
			time.Sleep(100 * time.Millisecond)
			if output != server.Jobs[jobID].Output {
				diff := server.Jobs[jobID].Output[len(output):]
				fmt.Printf(diff)
				output += diff
			}
			if server.Jobs[jobID].Status == "finished" {
				break
			}
		}
		os.Exit(0)
	}

	/*
		// tokenize the prompt
		embdInp := ml.Tokenize(ctx.Vocab, prompt, true)

		var embd []uint32

		// Initialize the ring buffer
		lastNTokens := ring.New(int(params.CtxSize))

		for i := 0; i < int(params.CtxSize); i++ {
			lastNTokens.Value = uint32(0)
			lastNTokens = lastNTokens.Next()
		}

		// A function to append a token to the ring buffer
		appendToken := func(token uint32) {
			lastNTokens.Value = token
			lastNTokens = lastNTokens.Next()
		}

		inputNoEcho := false
		pastCount := uint32(0)
		remainCount := params.PredictCount
		consumedCount := uint32(0)

		tokenCounter := 0
		evalPerformance := make([]int64, 0, params.PredictCount)
		fullPerformance := make([]int64, 0, params.PredictCount)
	*/ /*
			for remainCount != 0 || params.Interactive {

				start := time.Now().UnixNano()

				// --- predict

				if len(embd) > 0 {

					// infinite text generation via context swapping
					// if we run out of context:
					// - take the n_keep first tokens from the original prompt (via n_past)
					// - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch

					if pastCount+uint32(len(embd)) > params.CtxSize {
						leftCount := pastCount - params.KeepCount
						pastCount = params.KeepCount

						// insert n_left/2 tokens at the start of embd from last_n_tokens
						//embd = append(lastNTokens[:leftCount/2], embd...)
						embd = append(llama.ExtractTokens(lastNTokens.Move(-int(leftCount/2)), int(leftCount/2)), embd...)
					}

					evalStart := time.Now().UnixNano()
					if err := llama.Eval(ctx, embd, pastCount, params); err != nil {
						fmt.Printf("\n[ERROR] Failed to eval")
						os.Exit(1)
					}
					evalPerformance = append(evalPerformance, time.Now().UnixNano()-evalStart)
				}

				pastCount += uint32(len(embd))
				embd = []uint32{}

				if len(embdInp) <= int(consumedCount) { // && !isInteracting {

					if params.IgnoreEOS {
						ctx.Logits[ml.TOKEN_EOS] = 0
					}

						//id := llama.SampleTopPTopK(ctx,
						//	lastNTokens[params.ctxSize-params.repeatLastN:], params.repeatLastN,
						//	params.topK, params.topP, params.temp, params.repeatPenalty)
		                //
						//lastNTokens = lastNTokens[1:] ////last_n_tokens.erase(last_n_tokens.begin());
						//lastNTokens = append(lastNTokens, id)

					id := llama.SampleTopPTopK(ctx,
						lastNTokens, params.RepeatLastN,
						params.TopK, params.TopP, params.Temp, params.RepeatPenalty)

					appendToken(id)

					// replace end of text token with newline token when in interactive mode
					if id == ml.TOKEN_EOS && params.Interactive && !params.Instruct {
						id = ml.NewLineToken
					}

					// add it to the context
					embd = append(embd, id)

					// echo this to console
					inputNoEcho = false

					// decrement remaining sampling budget
					remainCount--

				} else {

					// some user input remains from prompt or interaction, forward it to processing

						//for len(embdInp) > int(consumedCount) {
						//	embd = append(embd, embdInp[consumedCount])
						//	if len(lastNTokens) > 0 {
						//		lastNTokens = lastNTokens[1:]
						//	}
						//	lastNTokens = append(lastNTokens, embdInp[consumedCount])
						//	consumedCount++
						//	if len(embd) >= int(params.batchSize) {
						//		break
						//	}
						//}

					for len(embdInp) > int(consumedCount) {
						embd = append(embd, embdInp[consumedCount])
						appendToken(embdInp[consumedCount])
						consumedCount++
						if len(embd) >= int(params.BatchSize) {
							break
						}
					}
				}

				// --- display text

				if !inputNoEcho {
					for _, id := range embd {

						token := ml.Token2Str(ctx.Vocab, id)
						final += token

						if len(strings.TrimSpace(final)) < len(strings.TrimSpace(prompt)) {
							continue
						}

						out := strings.Split(final, prompt)

						if len(out) == 2 && token == "\n" {
							continue
						}

						if len(strings.TrimSpace(final)) == len(strings.TrimSpace(prompt)) && (token != "\n") && (len(out) == 2) {
							Colorize("\n\n[magenta]▒▒▒ [light_yellow]" + strings.TrimSpace(prompt) + "\n[light_blue]▒▒▒ ")
							continue
						}

						Colorize("[white]" + token)

						tokenCounter++
						fullPerformance = append(fullPerformance, time.Now().UnixNano()-start)

						if ml.DEBUG {
							fmt.Printf(" [ #%d | %d ] ", tokenCounter, fullPerformance[len(fullPerformance)-1]/1_000_000)
						}
					}
				}
			}

			if ml.DEBUG {
				//Colorize("\n\n=== TOKEN EVAL TIMINGS ===\n\n")
				//for _, time := range evalPerformance {
				//	Colorize("%d | ", time/1_000_000)
				//}

				Colorize("\n\n=== FULL TIMINGS ===\n\n")
				for _, time := range fullPerformance {
					Colorize("%d | ", time/1_000_000)
				}
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
	*/
}

func parseOptions() *Options {

	var opts Options

	_, err := flags.Parse(&opts)
	if err != nil {
		Colorize("\n[magenta][ ERROR ][white] Can't parse options from command line!\n\n")
		os.Exit(0)
	}

	if opts.Model == "" {
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
		opts.Threads = runtime.NumCPU()
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
		opts.Temp = 0.5
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

	// https://patorjk.com/software/taag/#p=display&f=3-D&t=llama.go%0A%0ALLaMA.go
	// Isometric 1, Modular, Rectangles, Rozzo, Small Isometric 1, 3-D

	logo := `                                                    
  /88       /88         /888/888   /88/8888/88   /888/888      /8888/88   /888/888    
  /888      /888      /888/ /888 /888/8888/888 /888/ /888     /8888 //   /8888//888  
  /8888/88  /8888/88  /8888/8888 /888/8888/888 /8888/8888 /88 /8888/8888 /888 /8888 
  /8888/888 /8888/888 /888 /8888 /888//88 /888 /888 /8888 /888//8888/88  //888/888
  //// ///  //// ///  ///  ////  ///  //  ///  ///  ////  ///  //// //    /// ///`

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
		"\n\n   [magenta]▒▒▒▒[light_magenta] [ LLaMA.go v" +
			VERSION +
			" ] [light_blue][ LLaMA GPT in pure Golang - based on LLaMA C++ ] [magenta]▒▒▒▒\n\n")
}

func downloadModel(dir, model string) (int64, error) {

	url := "https://nogpu.com/" + model
	file := dir + "/" + model

	// TODO: check file existence first with io.IsExist
	output, err := os.Create(file)
	if err != nil {
		return 0, err
	}
	defer output.Close()

	response, err := http.Get(url)
	if err != nil {
		return 0, err
	}
	defer response.Body.Close()

	n, err := io.Copy(output, response.Body)
	if err != nil {
		return 0, err
	}

	if n < 1_000_000 {
		return 0, fmt.Errorf("some problem with target file")
	}

	return n, nil
}
