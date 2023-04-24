package server

import (
	"fmt"
	"time"

	fiber "github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

type Job struct {
	ID         string
	Status     string
	CreatedAt  int64
	StartedAt  int64
	FinishedAt int64
}

var (
	Host string
	Port string

	CtxSize uint32

	// TODO: sync.Map
	// TODO: Helicopter View - how to work with balancers and multi-pod architectures?
	Jobs  map[string]*Job // ID -> Job
	Queue map[string]*Job // ID -> Status
)

func init() {
	Jobs = make(map[string]*Job)
	Queue = make(map[string]*Job)
}

func Run() {

	app := fiber.New(fiber.Config{
		DisableStartupMessage: true,
	})

	//app.Get("/", func(c *fiber.Ctx) error {
	//	return c.SendString("Hello, World 👋!")
	//})
	app.Get("/jobs/status/:id", GetStatus)
	app.Post("/jobs/", NewJob)

	go Engine()

	app.Listen(Host + ":" + Port)
}

func Engine() {
	for {

		if len(Queue) == 0 {
			fmt.Printf(" [ SLEEP ] ")
			time.Sleep(1 * time.Second)
			continue
		}

		for id, job := range Queue {
			fmt.Printf("\n[ ENGINE ] Moving job id # %s from Queue to Jobs", id)
			Jobs[id].Status = "processing"
			delete(Queue, id)
			go Do(job)
		}
	}
}

func Do(job *Job) {
	Jobs[job.ID].StartedAt = time.Now().Unix()
	fmt.Printf("\n[ PROCESSING ] Starting job # %s", job.ID)
	time.Sleep(10 * time.Second)
	Jobs[job.ID].FinishedAt = time.Now().Unix()
	fmt.Printf("\n[ PROCESSING ] Finishing job # %s", job.ID)
	Jobs[job.ID].Status = "finished"

}

// POST /jobs
// {
//     "id": "",
//     "prompt": ""
// }

func NewJob(ctx *fiber.Ctx) error {

	//config := ctx.App().Config()
	//fmt.Printf("%+v", config)

	payload := struct {
		ID     string `json:"id"`
		Prompt string `json:"prompt"`
	}{}

	if err := ctx.BodyParser(&payload); err != nil {
		//return err
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
	if len(payload.Prompt) >= int(CtxSize) {
		return ctx.
			Status(fiber.StatusBadRequest).
			SendString(fmt.Sprintf("Prompt length %d is more than allowed %d chars!", len(payload.Prompt), CtxSize))
	}

	// TODO: Tokenize and check for max tokens

	Jobs[payload.ID] = &Job{
		ID:        payload.ID,
		Status:    "queued",
		CreatedAt: time.Now().Unix(),
	}

	Queue[payload.ID] = &Job{
		ID:        payload.ID,
		Status:    "queued",
		CreatedAt: time.Now().Unix(),
	}

	return ctx.JSON(fiber.Map{
		"id":       payload.ID,
		"created":  Jobs[payload.ID].CreatedAt,
		"started":  Jobs[payload.ID].StartedAt,
		"finished": "2023-04-24 13:47:00 GMT+00", // time.Now().Unix(),
		"model":    "mira-beta-7B",
		"source":   "web",
		"status":   Jobs[payload.ID].Status,
	})
}

// GET /jobs/status/:id

func GetStatus(ctx *fiber.Ctx) error {

	//config := ctx.App().Config()
	//fmt.Printf("%+v", config)

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

	return ctx.JSON(fiber.Map{
		"id":       ctx.Params("id"),
		"created":  Jobs[id].CreatedAt,
		"started":  Jobs[id].StartedAt,
		"finished": Jobs[id].FinishedAt,
		"model":    "mira-beta-7B",
		"status":   Jobs[ctx.Params("id")].Status,
	})
}

// GET /jobs/:id
