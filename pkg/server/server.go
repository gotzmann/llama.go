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
	Jobs  map[string]*Job   // ID -> Job
	Queue map[string]string // ID -> Status
)

func init() {
	Jobs = make(map[string]*Job)
	Queue = make(map[string]string)
}

func Run() {

	app := fiber.New(fiber.Config{
		DisableStartupMessage: true,
	})

	app.Get("/", func(c *fiber.Ctx) error {
		return c.SendString("Hello, World 👋!")
	})

	app.Get("/jobs/status/:id", GetStatus)

	app.Post("/jobs/", NewJob)

	app.Listen(Host + ":" + Port)
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

	return ctx.JSON(fiber.Map{
		"id":       payload.ID,
		"received": time.Now().Unix(),
		"started":  time.Now().Unix(),
		"finished": "2023-04-24 13:47:00 GMT+00", // time.Now().Unix(),
		"model":    "mira-beta-7B",
		"source":   "web",
		"status":   "processing",
	})
	//return c.SendString("Hello, World 👋!")
}

// GET /jobs/status/:id

func GetStatus(ctx *fiber.Ctx) error {

	//config := ctx.App().Config()
	//fmt.Printf("%+v", config)

	if _, err := uuid.Parse(ctx.Params("id")); err != nil {
		return ctx.
			Status(fiber.StatusBadRequest).
			SendString("Wrong UUID4 id for request!")
	}

	if _, ok := Jobs[ctx.Params("id")]; !ok {
		return ctx.
			Status(fiber.StatusBadRequest).
			SendString("Request ID was not found!")
	}

	return ctx.JSON(fiber.Map{
		"id":       ctx.Params("id"),
		"received": time.Now().Unix(),
		"started":  time.Now().Unix(),
		"finished": time.Now().Unix(),
		"model":    "mira-beta-7B",
		"status":   "processing",
	})
	//return c.SendString("Hello, World 👋!" + "/jobs/status/:id")
}

// GET /jobs/:id
