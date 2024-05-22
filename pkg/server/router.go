package server

import (
	fiber "github.com/gofiber/fiber/v2"
)

func WireRoutes(app *fiber.App) {

	// -- Booster Async API

	app.Post("/jobs/", NewJob)
	app.Delete("/jobs/:id", StopJob)
	app.Get("/jobs/status/:id", GetJobStatus)
	app.Get("/jobs/:id", GetJob)

	// -- OpenAI compatible API

	app.Post("/v1/chat/completions", NewChatCompletions)

	// -- Ollama compatible API

	app.Get("/api/tags", GetModels)

	app.Get("/api/version", func(ctx *fiber.Ctx) error {
		return ctx.
			JSON(fiber.Map{
				"version": "3.0.0", // TODO: booster.VERSION,
			})
	})

	// -- Monitoring Endpoints

	app.Get("/health", GetHealth)
}
