package main

import (
	"container/ring"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"

	flags "github.com/jessevdk/go-flags"
	colorable "github.com/mattn/go-colorable"
	"github.com/mitchellh/colorstring"
	"github.com/pkg/profile"

	"github.com/gotzmann/llama.go/pkg/llama"
	"github.com/gotzmann/llama.go/pkg/ml"
)

const VERSION = "1.2.0"

func main() {

	// --- Parse command line args and set default parameters

	var opts struct {
		Prompt  string  `long:"prompt" description:"Text prompt from user to feed the model input"`
		Model   string  `long:"model" description:"Path and file name of converted .bin LLaMA model"`
		Threads int     `long:"threads" description:"Adjust to the number of CPU cores you want to use [ all cores by default ]"`
		Predict uint32  `long:"predict" description:"Number of tokens to predict [ 64 by default ]"`
		Context uint32  `long:"context" description:"Context size in tokens [ 64 by default ]"`
		Temp    float32 `long:"temp" description:"Model temperature hyper parameter [ 0.80 by default ]"`
		Silent  bool    `long:"silent" description:"Hide welcome logo and other output [ show by default ]"`
		Chat    bool    `long:"chat" description:"Chat with user in interactive mode instead of compute over static prompt"`
		Profile bool    `long:"profile" description:"Profe CPU performance while running and store results to [cpu.pprof] file"`
		UseAVX  bool    `long:"avx" description:"Enable x64 AVX2 optimizations for Intel / AMD machines"`
		UseNEON bool    `long:"neon" description:"Enable ARM NEON optimizations for Apple / ARM machines"`
	}

	_, err := flags.Parse(&opts)
	if err != nil {
		return
	}

	if opts.Profile {
		defer profile.Start(profile.ProfilePath(".")).Stop()
	}

	prompt := " " + opts.Prompt // add a space to match LLaMA tokenizer behavior
	final := ""                 // accumulate model output

	// Allow to use ALL cores for the program itself and CLI specified number of cores for the parallel tensor math
	// TODO Optimize default settings for CPUs with P and E cores like M1 Pro = 8 performant and 2 energy cores
	// runtime.GOMAXPROCS(runtime.NumCPU())
	if opts.Threads == 0 {
		opts.Threads = runtime.NumCPU()
	}

	if opts.Context == 0 {
		opts.Context = 64
	}

	if opts.Predict == 0 {
		opts.Predict = 64
	}

	if opts.Temp == 0 {
		opts.Temp = 0.8
	}

	repeatLastN := uint32(64)
	if repeatLastN > opts.Context {
		repeatLastN = opts.Context
	}

	if !opts.Silent {
		showLogo()
	}

	if opts.Prompt == "" || opts.Model == "" {
		Colorize("\n[magenta][ ERROR ][white] Please specify correct model path and prompt with [light_magenta]--model[white] and [light_magenta]--prompt[white] parameters\n\n")
		os.Exit(0)
	}

	params := llama.ModelParams{
		Model: opts.Model,

		MaxThreads: opts.Threads,

		UseAVX:  opts.UseAVX,
		UseNEON: opts.UseNEON,

		Interactive: opts.Chat,

		CtxSize:      opts.Context,
		Seed:         -1,
		PredictCount: opts.Predict,
		RepeatLastN:  repeatLastN,
		PartsCount:   -1,
		BatchSize:    8,

		TopK:          40,
		TopP:          0.95,
		Temp:          opts.Temp,
		RepeatPenalty: 1.10,

		MemoryFP16: true,
	}

	// --- load the model

	ctx, err := llama.LoadModel(params.Model, opts.Silent)
	if err != nil {
		Colorize("\n[magenta][ ERROR ][white] Failed to load model [light_magenta]\"%s\"\n\n", params.Model)
		os.Exit(0)
	}

	// tokenize the prompt
	embdInp := ml.Tokenize(ctx.Vocab, prompt, true)
	tokenNewline := ml.Tokenize(ctx.Vocab, "\n", false)[0]

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
	evalPerformance := make([]int64, 0, opts.Predict)

	for remainCount != 0 || params.Interactive {

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

			start := time.Now().UnixNano()
			if err := llama.Eval(ctx, embd, uint32(len(embd)), pastCount, params); err != nil {
				fmt.Printf("\n[ERROR] Failed to eval")
				os.Exit(1)
			}
			evalPerformance = append(evalPerformance, time.Now().UnixNano()-start)
		}

		pastCount += uint32(len(embd))
		embd = []uint32{}

		if len(embdInp) <= int(consumedCount) { // && !isInteracting {

			if params.IgnoreEOS {
				ctx.Logits[ml.TOKEN_EOS] = 0
			}

			/*
				id := llama.SampleTopPTopK(ctx,
					lastNTokens[params.ctxSize-params.repeatLastN:], params.repeatLastN,
					params.topK, params.topP, params.temp, params.repeatPenalty)

				lastNTokens = lastNTokens[1:] ////last_n_tokens.erase(last_n_tokens.begin());
				lastNTokens = append(lastNTokens, id)

			*/
			id := llama.SampleTopPTopK(ctx,
				lastNTokens, params.RepeatLastN,
				params.TopK, params.TopP, params.Temp, params.RepeatPenalty)

			appendToken(id)

			// replace end of text token with newline token when in interactive mode
			if id == ml.TOKEN_EOS && params.Interactive && !params.Instruct {
				id = tokenNewline
			}

			// add it to the context
			embd = append(embd, id)

			// echo this to console
			inputNoEcho = false

			// decrement remaining sampling budget
			remainCount--

		} else {

			// some user input remains from prompt or interaction, forward it to processing
			/*
				for len(embdInp) > int(consumedCount) {
					embd = append(embd, embdInp[consumedCount])
					if len(lastNTokens) > 0 {
						lastNTokens = lastNTokens[1:]
					}
					lastNTokens = append(lastNTokens, embdInp[consumedCount])
					consumedCount++
					if len(embd) >= int(params.batchSize) {
						break
					}
				}
			*/
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
			}
		}
	}

	if ml.DEBUG {
		Colorize("\n\n=== TOKEN EVAL TIMINGS ===\n\n")
		for _, time := range evalPerformance {
			Colorize("%d | ", time/1_000_000)
		}
	}

	avgEval := int64(0)
	for _, time := range evalPerformance {
		avgEval += time / 1_000_000
	}
	avgEval /= int64(len(evalPerformance))

	Colorize(
		"\n\n[light_magenta][ HALT ][white] Time per token: [light_cyan]%d[white] ms | Tokens per second: [light_cyan]%.2f\n\n",
		avgEval,
		float64(1000)/float64(avgEval))
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
