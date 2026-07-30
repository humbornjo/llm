// Example: Multi-provider usage
//
// This example demonstrates how to use multiple providers with the same code.
//
// Run with:
//
//	export OPENAI_API_KEY="sk-..."
//	export ANTHROPIC_API_KEY="sk-ant-..."
//	go run main.go
package main

import (
	"context"
	"errors"
	"fmt"

	"github.com/humbornjo/llm"
	"github.com/humbornjo/llm/providers/anthropic"
	"github.com/humbornjo/llm/providers/openai"
)

func main() {
	ctx := context.Background()

	prompt := "What is 2 + 2? Reply with just the number."
	fmt.Printf("Prompt: %s\n\n", prompt)

	// Try OpenAI.
	fmt.Println("OpenAI:")
	if err := tryProvider(ctx, "openai", "gpt-4o-mini", prompt); err != nil {
		fmt.Printf("  Error: %v\n\n", err)
	}

	// Try Anthropic.
	fmt.Println("Anthropic:")
	if err := tryProvider(ctx, "anthropic", "claude-3-5-haiku-latest", prompt); err != nil {
		fmt.Printf("  Error: %v\n\n", err)
	}
}

func tryProvider(ctx context.Context, providerName, model, prompt string) error {
	var provider llm.Provider
	var err error

	switch providerName {
	case "openai":
		provider, err = openai.New()
	case "anthropic":
		provider, err = anthropic.New()
	default:
		return fmt.Errorf("unknown provider: %s", providerName)
	}

	if err != nil {
		if errors.Is(err, llm.ErrMissingAPIKey) {
			fmt.Printf("  Skipped: API key not configured\n\n")
			return nil
		}
		return err
	}

	response, err := provider.Completion(ctx, llm.CompletionParams{
		Model: model,
		Messages: []llm.Message{
			{Role: llm.ROLE_USER, Content: llm.ContentFromString(prompt)},
		},
	})
	if err != nil {
		return err
	}

	fmt.Printf("  Response: %s\n", response.Choices[0].Message.Content)
	fmt.Printf("  Tokens: %d\n\n", response.Usage.TotalTokens)
	return nil
}
