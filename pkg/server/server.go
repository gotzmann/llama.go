package server

import (
	"container/ring"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	fiber "github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	colorable "github.com/mattn/go-colorable"
	"github.com/mitchellh/colorstring"

	"github.com/gotzmann/llama.go/pkg/llama"
	"github.com/gotzmann/llama.go/pkg/ml"
)

// TODO: Helicopter View - how to work with balancers and multi-pod architectures?
// TODO: Rate Limiter based on end-user IP address
// TODO: Guard access with API Tokens

// Unix timestamps VS ISO-8601 Stripe perspective:
// https://dev.to/stripe/how-stripe-designs-for-dates-and-times-in-the-api-3eoh
// TODO: UUID vs string for job ID
// TODO: Unix timestamp vs ISO for date and time

type Job struct {
	ID         string
	Status     string
	Prompt     string
	Output     string
	CreatedAt  int64
	StartedAt  int64
	FinishedAt int64
}

var (
	Host string
	Port string

	Vocab *ml.Vocab
	Model *llama.Model

	Ctx    *llama.Context
	Params *llama.ModelParams

	MaxPods     int64
	RunningPods int64 // number of pods running right now in parallel

	mu sync.Mutex // guards any Jobs change

	// TODO: Background watcher which will make waiting jobs obsolete after some deadline
	Jobs  map[string]*Job     // all seen jobs in any state
	Queue map[string]struct{} // queue of job IDs waiting for start
)

func init() {
	Jobs = make(map[string]*Job)
	Queue = make(map[string]struct{})
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

		for jobID, _ := range Queue {

			if RunningPods >= MaxPods {
				continue
			}

			mu.Lock()
			Jobs[jobID].Status = "processing"
			delete(Queue, jobID)
			mu.Unlock()

			atomic.AddInt64(&RunningPods, 1)
			go Do(jobID)
		}

		// TODO: Some internal sync with channels, not time.Sleep()
		time.Sleep(1000 * time.Millisecond)
	}
}

// --- worker doing the "job" of transforming boring prompt into magic output

func Do(jobID string) {

	defer atomic.AddInt64(&RunningPods, -1)
	defer runtime.GC()

	// TODO: Proper logging
	// fmt.Printf("\n[ PROCESSING ] Starting job # %s", jobID)

	mu.Lock()
	Jobs[jobID].StartedAt = time.Now().Unix()
	prompt := " " + Jobs[jobID].Prompt // add a space to match LLaMA tokenizer behavior
	mu.Unlock()

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

// --- Place new job into queue

func PlaceJob(jobID, prompt string) {

	timing := time.Now().Unix()

	mu.Lock()

	Jobs[jobID] = &Job{
		ID:        jobID,
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
		ID     string `json:"id"`
		Prompt string `json:"prompt"`
	}{}

	if err := ctx.BodyParser(&payload); err != nil {
		// TODO: Proper error handling
	}

	if _, err := uuid.Parse(payload.ID); err != nil {
		return ctx.
			Status(fiber.StatusBadRequest).
			SendString("Wrong UUID4 id for request!")
	}

	if _, ok := Jobs[payload.ID]; ok {
		return ctx.
			Status(fiber.StatusBadRequest).
			SendString("Duplicated ID for the same request?")
	}

	// TODO: Proper chack for max chars in request
	if len(payload.Prompt) >= int(Params.CtxSize) {
		return ctx.
			Status(fiber.StatusBadRequest).
			SendString(fmt.Sprintf("Prompt length %d is more than allowed %d chars!", len(payload.Prompt), Params.CtxSize))
	}

	// TODO: Tokenize and check for max tokens

	PlaceJob(payload.ID, payload.Prompt)

	// TODO: Guard with mutex
	return ctx.JSON(fiber.Map{
		"id":      payload.ID,
		"prompt":  payload.Prompt,
		"created": Jobs[payload.ID].CreatedAt,
		//"started":  Jobs[payload.ID].StartedAt,
		//"finished": Jobs[payload.ID].FinishedAt,
		//"model":    "model-xx", // TODO: Real model ID
		//"source":   "api",      // TODO: Enum for sources
		"status": Jobs[payload.ID].Status,
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
