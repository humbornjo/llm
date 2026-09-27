// Example: Streaming responses
//
// This example demonstrates how to receive streaming responses.
//
// Run with:
//
//	export OPENAI_API_KEY="sk-..."
//	go run main.go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/humbornjo/llm"
	"github.com/humbornjo/llm/providers/openai"
)

func main() {
	// Create a provider instance for better performance with multiple requests.
	provider, err := openai.New()
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	// Request a streaming completion.
	chunks := provider.CompletionStream(ctx, llm.CompletionParams{
		Model: "gpt-4o-mini",
		Messages: []llm.Message{
			{Role: llm.ROLE_USER, Content: llm.ContentFromString("Write a short poem about programming in Go.")},
		},
		Stream: true,
	})

	fmt.Println("Streaming response:")
	fmt.Println("---")

	// Process chunks as they arrive.
	for chunk, err := range chunks {
		if err != nil {
			log.Fatal(err)
		}
		if len(chunk.Choices) > 0 {
			content := chunk.Choices[0].Delta.Content
			if content != "" {
				fmt.Print(content)
			}
		}
	}

	fmt.Println("\n---")

	fmt.Println("Stream completed successfully!")
}
