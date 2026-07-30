package testutil

import (
	"context"
	"iter"

	"github.com/humbornjo/llm/providers"
)

// MockProvider is a mock implementation of the Provider interface for testing.
type MockProvider struct {
	NameFunc             func() string
	CompletionFunc       func(ctx context.Context, params providers.CompletionParams) (*providers.ChatCompletion, error)
	CompletionStreamFunc func(ctx context.Context, params providers.CompletionParams) iter.Seq2[providers.ChatCompletionChunk, error]
	EmbeddingFunc        func(ctx context.Context, params providers.EmbeddingParams) (*providers.EmbeddingResponse, error)
	ListModelsFunc       func(ctx context.Context) (*providers.ModelsResponse, error)
	CapabilitiesFunc     func() providers.Capabilities

	// Track calls for assertions.
	CompletionCalls       []providers.CompletionParams
	CompletionStreamCalls []providers.CompletionParams
	EmbeddingCalls        []providers.EmbeddingParams
	ListModelsCalls       int
}

// Ensure MockProvider implements all interfaces.
var (
	_ providers.Provider           = (*MockProvider)(nil)
	_ providers.EmbeddingProvider  = (*MockProvider)(nil)
	_ providers.ModelLister        = (*MockProvider)(nil)
	_ providers.CapabilityProvider = (*MockProvider)(nil)
)

// NewMockProvider creates a new MockProvider with default implementations.
func NewMockProvider() *MockProvider {
	return &MockProvider{
		NameFunc: func() string { return "mock" },
		CompletionFunc: func(ctx context.Context, params providers.CompletionParams) (*providers.ChatCompletion, error) {
			return &providers.ChatCompletion{
				ID:     "mock-completion-id",
				Object: "chat.completion",
				Model:  params.Model,
				Choices: []providers.Choice{
					{
						Index: 0,
						Message: providers.Message{
							Role:    providers.ROLE_ASSISTANT,
							Content: providers.ContentFromString("Hello World"),
						},
						FinishReason: providers.FINISH_REASON_STOP,
					},
				},
				Usage: &providers.Usage{
					PromptTokens:     10,
					CompletionTokens: 5,
					TotalTokens:      15,
				},
			}, nil
		},
		CompletionStreamFunc: func(ctx context.Context, params providers.CompletionParams) iter.Seq2[providers.ChatCompletionChunk, error] {
			return func(yield func(providers.ChatCompletionChunk, error) bool) {
				chunks := []providers.ChatCompletionChunk{
					{
						ID:     "mock-chunk-id",
						Object: "chat.completion.chunk",
						Model:  params.Model,
						Choices: []providers.ChunkChoice{
							{Index: 0, Delta: providers.ChunkDelta{Role: providers.ROLE_ASSISTANT}},
						},
					},
					{
						ID:     "mock-chunk-id",
						Object: "chat.completion.chunk",
						Model:  params.Model,
						Choices: []providers.ChunkChoice{
							{Index: 0, Delta: providers.ChunkDelta{Content: "Hello World"}},
						},
					},
					{
						ID:     "mock-chunk-id",
						Object: "chat.completion.chunk",
						Model:  params.Model,
						Choices: []providers.ChunkChoice{
							{Index: 0, FinishReason: providers.FINISH_REASON_STOP},
						},
					},
				}
				for _, chunk := range chunks {
					if err := ctx.Err(); err != nil {
						yield(providers.ChatCompletionChunk{}, err)
						return
					}
					if !yield(chunk, nil) {
						return
					}
				}
			}
		},
		EmbeddingFunc: func(ctx context.Context, params providers.EmbeddingParams) (*providers.EmbeddingResponse, error) {
			return &providers.EmbeddingResponse{
				Object: "list",
				Model:  params.Model,
				Data: []providers.EmbeddingData{
					{
						Object:    "embedding",
						Embedding: []float64{0.1, 0.2, 0.3},
						Index:     0,
					},
				},
				Usage: &providers.EmbeddingUsage{
					PromptTokens: 5,
					TotalTokens:  5,
				},
			}, nil
		},
		ListModelsFunc: func(ctx context.Context) (*providers.ModelsResponse, error) {
			return &providers.ModelsResponse{
				Object: "list",
				Data: []providers.Model{
					{ID: "model-1", Object: "model", OwnedBy: "mock"},
					{ID: "model-2", Object: "model", OwnedBy: "mock"},
				},
			}, nil
		},
		CapabilitiesFunc: func() providers.Capabilities {
			return providers.Capabilities{
				Completion:          true,
				CompletionStreaming: true,
				CompletionTools:     true,
				Embedding:           true,
				ListModels:          true,
			}
		},
	}
}

func (m *MockProvider) Name() string {
	return m.NameFunc()
}

func (m *MockProvider) Completion(
	ctx context.Context,
	params providers.CompletionParams,
) (*providers.ChatCompletion, error) {
	m.CompletionCalls = append(m.CompletionCalls, params)
	return m.CompletionFunc(ctx, params)
}

func (m *MockProvider) CompletionStream(
	ctx context.Context,
	params providers.CompletionParams,
) iter.Seq2[providers.ChatCompletionChunk, error] {
	m.CompletionStreamCalls = append(m.CompletionStreamCalls, params)
	return m.CompletionStreamFunc(ctx, params)
}

func (m *MockProvider) Embedding(
	ctx context.Context,
	params providers.EmbeddingParams,
) (*providers.EmbeddingResponse, error) {
	m.EmbeddingCalls = append(m.EmbeddingCalls, params)
	return m.EmbeddingFunc(ctx, params)
}

func (m *MockProvider) ListModels(ctx context.Context) (*providers.ModelsResponse, error) {
	m.ListModelsCalls++
	return m.ListModelsFunc(ctx)
}

func (m *MockProvider) Capabilities() providers.Capabilities {
	return m.CapabilitiesFunc()
}

// MockChatCompletion creates a mock ChatCompletion response.
func MockChatCompletion(content string) *providers.ChatCompletion {
	return &providers.ChatCompletion{
		ID:     "mock-id",
		Object: "chat.completion",
		Model:  "mock-model",
		Choices: []providers.Choice{
			{
				Index: 0,
				Message: providers.Message{
					Role:    providers.ROLE_ASSISTANT,
					Content: providers.ContentFromString(content),
				},
				FinishReason: providers.FINISH_REASON_STOP,
			},
		},
		Usage: &providers.Usage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}
}

// MockChatCompletionWithToolCalls creates a mock ChatCompletion with tool calls.
func MockChatCompletionWithToolCalls(toolCalls []providers.ToolCall) *providers.ChatCompletion {
	return &providers.ChatCompletion{
		ID:     "mock-id",
		Object: "chat.completion",
		Model:  "mock-model",
		Choices: []providers.Choice{
			{
				Index: 0,
				Message: providers.Message{
					Role:      providers.ROLE_ASSISTANT,
					Content:   providers.ContentFromString(""),
					ToolCalls: toolCalls,
				},
				FinishReason: providers.FINISH_REASON_TOOL_CALLS,
			},
		},
		Usage: &providers.Usage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
	}
}

// MockChatCompletionWithReasoning creates a mock ChatCompletion with reasoning.
func MockChatCompletionWithReasoning(content, reasoning string) *providers.ChatCompletion {
	return &providers.ChatCompletion{
		ID:     "mock-id",
		Object: "chat.completion",
		Model:  "mock-model",
		Choices: []providers.Choice{
			{
				Index: 0,
				Message: providers.Message{
					Role:    providers.ROLE_ASSISTANT,
					Content: providers.ContentFromString(content),
					Reasoning: &providers.Reasoning{
						Content: reasoning,
					},
				},
				FinishReason: providers.FINISH_REASON_STOP,
			},
		},
		Usage: &providers.Usage{
			PromptTokens:     10,
			CompletionTokens: 50,
			TotalTokens:      60,
			ReasoningTokens:  30,
		},
	}
}
