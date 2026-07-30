// Example: Tool/Function calling
//
// This example demonstrates how to use tools (function calling) with llm.
//
// Run with:
//
//	export OPENAI_API_KEY="sk-..."
//	go run main.go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/humbornjo/llm"
	"github.com/humbornjo/llm/providers/openai"
)

// Define tools that the model can call.
var tools = []llm.ToolInfo{
	{
		Type: "function",
		Function: llm.Function{
			Name:        "get_weather",
			Description: "Get the current weather for a location",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{
						"type":        "string",
						"description": "The city name, e.g., 'Paris' or 'New York'",
					},
					"unit": map[string]any{
						"type":        "string",
						"enum":        []string{"celsius", "fahrenheit"},
						"description": "Temperature unit",
					},
				},
				"required": []string{"location"},
			},
		},
	},
}

// simulateWeather simulates a weather API call.
func simulateWeather(location, unit string) string {
	if unit == "" {
		unit = "celsius"
	}
	// In a real app, this would call an actual weather API.
	return fmt.Sprintf(`{"location": "%s", "temperature": 22, "unit": "%s", "condition": "sunny"}`, location, unit)
}

func main() {
	provider, err := openai.New()
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	// Initial message asking about weather.
	messages := []llm.Message{
		{Role: llm.ROLE_USER, Content: llm.ContentFromString("What's the weather like in Paris?")},
	}

	fmt.Println("User: What's the weather like in Paris?")
	fmt.Println()

	// First request - model may call the tool.
	response, err := provider.Completion(ctx, llm.CompletionParams{
		Model:      "gpt-4o-mini",
		Messages:   messages,
		Tools:      tools,
		ToolChoice: "auto",
	})
	if err != nil {
		log.Fatal(err)
	}

	// Check if the model wants to call a tool.
	if response.Choices[0].FinishReason == llm.FINISH_REASON_TOOL_CALLS {
		fmt.Println("Model is calling tools...")

		// Add the assistant's message (with tool calls) to the conversation.
		messages = append(messages, response.Choices[0].Message)

		// Process each tool call.
		for _, tc := range response.Choices[0].Message.ToolCalls {
			fmt.Printf("  Tool: %s\n", tc.Function.Name)
			fmt.Printf("  Arguments: %s\n", tc.Function.Arguments)

			// Parse the arguments.
			var args struct {
				Location string `json:"location"`
				Unit     string `json:"unit"`
			}
			if unmarshalErr := json.Unmarshal([]byte(tc.Function.Arguments), &args); unmarshalErr != nil {
				log.Fatal(unmarshalErr)
			}

			// Execute the function.
			result := simulateWeather(args.Location, args.Unit)
			fmt.Printf("  Result: %s\n", result)
			fmt.Println()

			// Add the tool result to the conversation.
			messages = append(messages, llm.Message{
				Role:       llm.ROLE_TOOL,
				Content:    llm.ContentFromString(result),
				ToolCallID: tc.ID,
			})
		}

		// Continue the conversation with the tool results.
		response, err = provider.Completion(ctx, llm.CompletionParams{
			Model:    "gpt-4o-mini",
			Messages: messages,
			Tools:    tools,
		})
		if err != nil {
			log.Fatal(err)
		}
	}

	// Print the final response.
	fmt.Printf("Assistant: %s\n", response.Choices[0].Message.Content)
}
