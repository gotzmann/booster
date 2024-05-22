package server

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
	"encoding/json"
	"reflect"
	"strings"
	"time"

	fiber "github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/valyala/fasthttp"
)

type Chunk struct {
	Message *CompletionMessage `json:"message"`

	Model              string    `json:"model,omitempty"`
	CreatedAt          time.Time `json:"created_at,omitempty"`
	Done               bool      `json:"done"`
	DoneReason         string    `json:"done_reason,omitempty"`
	TotalDuration      int       `json:"total_duration,omitempty"`
	LoadDuration       int       `json:"load_duration,omitempty"`
	PromptEvalDuration int       `json:"prompt_eval_duration,omitempty"`
	EvalCount          int       `json:"eval_count,omitempty"`
	EvalDuration       int       `json:"eval_duration,omitempty"`
}

func WireRoutes(app *fiber.App) {

	// -- Booster Async API

	app.Post("/jobs/", NewJob)
	app.Delete("/jobs/:id", StopJob)
	app.Get("/jobs/status/:id", GetJobStatus)
	app.Get("/jobs/:id", GetJob)

	// -- OpenAI compatible API

	app.Post("/v1/chat/completions", NewChatCompletions)

	// -- Ollama compatible API
	//    https://github.com/ollama/ollama/blob/main/docs/api.md

	app.Get("/api/version",
		func(ctx *fiber.Ctx) error {
			return ctx.JSON(
				fiber.Map{
					"version": "3.0.0", // TODO: booster.VERSION,
				})
		})

	app.Get("/api/tags",
		func(ctx *fiber.Ctx) error {
			models := make([]fiber.Map, 0)

			for name := range Models {
				models = append(models, fiber.Map{
					"name": name,
					// "modified_at": "2023-11-04T14:56:49.277302595-07:00",
					// "size": 7365960935,
					// "digest": "9f438cb9cd581fc025612d27f7c1a6669ff83a8bb0ed86c94fcf4c5440555697",
					// "details": {
					// "format": "gguf",
					// "family": "llama",
					// "families": null,
					// "parameter_size": "13B",
					// "quantization_level": "Q4_0"
				})
			}

			return ctx.JSON(
				fiber.Map{
					"models": models,
				})
		})

	// -- streaming endpoint !!!

	app.Post("/api/chat",
		func(ctx *fiber.Ctx) error {

			ctx.Set("Content-Type", "application/x-ndjson")

			if GoShutdown {
				return ctx.
					Status(fiber.StatusServiceUnavailable).
					JSON(fiber.Map{"error": "service is shutting down"})
			}

			payload := &CompletionPayload{}
			if err := json.Unmarshal(ctx.Body(), payload); err != nil {
				return ctx.
					Status(fiber.StatusBadRequest).
					JSON(fiber.Map{"error": "error parsing request body"})
			}

			sessionID := uuid.New().String()
			jobID := uuid.New().String()
			promptID := reflect.ValueOf(Prompts).MapKeys()[0].String()             // FIXME: using ANY available prompt for a while
			Sessions[sessionID], _ = buildCompletion(sessionID, promptID, payload) // TODO: error handling
			PlaceJob(jobID, "" /* payload.Model */, sessionID, "" /* prompt */)

			ctx.Context().SetBodyStreamWriter(
				fasthttp.StreamWriter(
					func(w *bufio.Writer) {

						prevOutput := ""
						for {

							time.Sleep(1 * time.Second)
							Mutex.Lock()

							output := C.GoString(C.status(C.CString(jobID)))
							// waiting while prompt history will be processed completely
							if Jobs[jobID].Status == "processing" && len(output) < len(Jobs[jobID].FullPrompt) {
								Mutex.Unlock()
								continue
							}
							output, _ = strings.CutPrefix(output, Jobs[jobID].FullPrompt)

							if Jobs[jobID].Status == "finished" {
								assistantTemplate := Prompts[Jobs[jobID].PromptID].Templates.Assistant
								if strings.Contains(assistantTemplate, "{ASSISTANT}") {
									cut := strings.Index(assistantTemplate, "{ASSISTANT}") + len("{ASSISTANT}")
									assistantSuffix := assistantTemplate[cut:]
									output, _ = strings.CutSuffix(output, assistantSuffix)
								}
							}

							Mutex.Unlock()

							if len(output) > len(prevOutput) {
								chunk := Chunk{
									Model: "hermes",
									Message: &CompletionMessage{
										Role:    "assistant",
										Content: output[len(prevOutput):],
									},
									CreatedAt: time.Now(),
									Done:      false,
								}

								json, _ := json.Marshal(chunk)
								json = append(json, '\n')
								w.Write(json)
								w.Flush()

								prevOutput = output
							}

							if Jobs[jobID].Status == "finished" {
								chunk := Chunk{
									Message: &CompletionMessage{
										Role:    "assistant",
										Content: "",
									},
									CreatedAt:  time.Now(),
									Done:       true,
									DoneReason: "stop",
								}

								json, _ := json.Marshal(chunk)
								w.Write(json)
								w.Flush()

								break
							}
						}
					}))

			return nil
		})

	// -- Monitoring Endpoints

	app.Get("/health", GetHealth)
}
