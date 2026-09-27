package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	stderrors "errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/genai"

	"github.com/humbornjo/llm/config"
	"github.com/humbornjo/llm/errors"
	"github.com/humbornjo/llm/internal/testutil"
	"github.com/humbornjo/llm/providers"
)

func TestGemini_New(t *testing.T) {
	t.Run("creates provider with API key", func(t *testing.T) {
		provider, err := New(config.WithAPIKey("test-api-key"))
		require.NoError(t, err)
		require.NotNil(t, provider)
		require.Equal(t, _PROVIDER_NAME, provider.Name())
	})

	t.Run("creates provider from GEMINI_API_KEY", func(t *testing.T) {
		t.Setenv("GEMINI_API_KEY", "env-api-key")

		provider, err := New()
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("creates provider from GOOGLE_API_KEY fallback", func(t *testing.T) {
		t.Setenv("GEMINI_API_KEY", "")
		t.Setenv("GOOGLE_API_KEY", "google-api-key")

		provider, err := New()
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("returns error when API key is missing", func(t *testing.T) {
		t.Setenv("GEMINI_API_KEY", "")
		t.Setenv("GOOGLE_API_KEY", "")

		provider, err := New()
		require.Nil(t, provider)
		require.Error(t, err)

		var missingKeyErr *errors.MissingAPIKeyError
		require.ErrorAs(t, err, &missingKeyErr)
		require.Equal(t, _PROVIDER_NAME, missingKeyErr.Provider)
		require.Equal(t, _ENV_API_KEY, missingKeyErr.EnvVar)
	})
}

func TestGemini_Capabilities(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithAPIKey("test-key"))
	require.NoError(t, err)

	caps := provider.Capabilities()

	require.True(t, caps.Completion)
	require.True(t, caps.CompletionReasoning)
	require.True(t, caps.CompletionStreaming)
	require.True(t, caps.CompletionTools)
	require.ElementsMatch(t, []providers.ContentPartType{
		providers.CONTENT_PART_TEXT,
		providers.CONTENT_PART_IMAGE_URL,
	}, caps.CompletionTypes)
	require.True(t, caps.Embedding)
	require.True(t, caps.ListModels)
}

func TestGemini_ConvertMessages(t *testing.T) {
	t.Parallel()

	t.Run("extracts system message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.ROLE_SYSTEM, Content: providers.ContentFromString("You are a helpful assistant.")},
			{Role: providers.ROLE_USER, Content: providers.ContentFromString("Hello")},
		}

		result, system := convertMessages(messages)

		require.NotNil(t, system)
		require.Len(t, system.Parts, 1)
		require.Equal(t, "You are a helpful assistant.", system.Parts[0].Text)
		require.Len(t, result, 1)
	})

	t.Run("concatenates multiple system messages", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.ROLE_SYSTEM, Content: providers.ContentFromString("First part.")},
			{Role: providers.ROLE_SYSTEM, Content: providers.ContentFromString("Second part.")},
			{Role: providers.ROLE_USER, Content: providers.ContentFromString("Hello")},
		}

		result, system := convertMessages(messages)

		require.NotNil(t, system)
		require.Contains(t, system.Parts[0].Text, "First part.")
		require.Contains(t, system.Parts[0].Text, "Second part.")
		require.Len(t, result, 1)
	})

	t.Run("converts user message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.ROLE_USER, Content: providers.ContentFromString("Hello")},
		}

		result, system := convertMessages(messages)

		require.Nil(t, system)
		require.Len(t, result, 1)
		require.Equal(t, "user", result[0].Role)
	})

	t.Run("converts assistant message to model role", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.ROLE_USER, Content: providers.ContentFromString("Hello")},
			{Role: providers.ROLE_ASSISTANT, Content: providers.ContentFromString("Hi there!")},
		}

		result, system := convertMessages(messages)

		require.Nil(t, system)
		require.Len(t, result, 2)
		require.Equal(t, _ROLE_MODEL, result[1].Role)
		require.Equal(t, "Hi there!", result[1].Parts[0].Text)
	})

	t.Run("converts assistant message with tool calls", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.ROLE_USER, Content: providers.ContentFromString("What's the weather?")},
			{
				Role:    providers.ROLE_ASSISTANT,
				Content: providers.ContentFromString(""),
				ToolCalls: []providers.ToolCall{
					{
						ID:   "call_123",
						Type: "function",
						Function: providers.FunctionCall{
							Name:      "get_weather",
							Arguments: `{"location": "Paris"}`,
						},
					},
				},
			},
		}

		result, _ := convertMessages(messages)

		require.Len(t, result, 2)
		require.Equal(t, _ROLE_MODEL, result[1].Role)
		require.NotNil(t, result[1].Parts[0].FunctionCall)
		require.Equal(t, "get_weather", result[1].Parts[0].FunctionCall.Name)
	})

	t.Run("converts tool result message with plain text", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.ROLE_USER, Content: providers.ContentFromString("Hello")},
			{Role: providers.ROLE_TOOL, Content: providers.ContentFromString("sunny, 22°C"), Name: "get_weather"},
		}

		result, _ := convertMessages(messages)

		require.Len(t, result, 2)
		require.Equal(t, "user", result[1].Role)
		require.NotNil(t, result[1].Parts[0].FunctionResponse)
		require.Equal(t, "get_weather", result[1].Parts[0].FunctionResponse.Name)
		// Plain text is wrapped as {"result": "sunny, 22°C"}.
		require.Equal(t, "sunny, 22°C", result[1].Parts[0].FunctionResponse.Response["result"])
	})

	t.Run("converts tool result message with JSON content", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.ROLE_USER, Content: providers.ContentFromString("Hello")},
			{
				Role:    providers.ROLE_TOOL,
				Content: providers.ContentFromString(`{"temperature": 22, "condition": "sunny"}`),
				Name:    "get_weather",
			},
		}

		result, _ := convertMessages(messages)

		require.Len(t, result, 2)
		require.NotNil(t, result[1].Parts[0].FunctionResponse)
		// JSON content is parsed directly.
		require.Equal(t, "sunny", result[1].Parts[0].FunctionResponse.Response["condition"])
	})

	t.Run("converts tool result message with fallback name", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.ROLE_TOOL, Content: providers.ContentFromString("result data")},
		}

		result, _ := convertMessages(messages)

		require.Len(t, result, 1)
		require.Equal(t, "function", result[0].Parts[0].FunctionResponse.Name)
	})

	t.Run("no system returns nil instruction", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.ROLE_USER, Content: providers.ContentFromString("Hello")},
		}

		_, system := convertMessages(messages)
		require.Nil(t, system)
	})

	t.Run("replays thought signature from Extra", func(t *testing.T) {
		t.Parallel()

		sig := base64.StdEncoding.EncodeToString([]byte("real-signature"))
		messages := []providers.Message{
			{Role: providers.ROLE_USER, Content: providers.ContentFromString("What's the weather?")},
			{
				Role:    providers.ROLE_ASSISTANT,
				Content: providers.ContentFromString(""),
				ToolCalls: []providers.ToolCall{
					{
						ID:   "call_123",
						Type: "function",
						Function: providers.FunctionCall{
							Name:      "get_weather",
							Arguments: `{"location": "Paris"}`,
						},
						Extra: map[string]providers.ProviderData{
							_PROVIDER_NAME: {_EXTRA_KEY_THOUGHT_SIGNATURE: sig},
						},
					},
				},
			},
		}

		result, _ := convertMessages(messages)

		require.Len(t, result, 2)
		require.Equal(t, _ROLE_MODEL, result[1].Role)
		require.NotNil(t, result[1].Parts[0].FunctionCall)
		require.Equal(t, []byte("real-signature"), result[1].Parts[0].ThoughtSignature)
	})

	t.Run("tool call without signature uses bypass value", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.ROLE_USER, Content: providers.ContentFromString("What's the weather?")},
			{
				Role:    providers.ROLE_ASSISTANT,
				Content: providers.ContentFromString(""),
				ToolCalls: []providers.ToolCall{
					{
						ID:   "call_123",
						Type: "function",
						Function: providers.FunctionCall{
							Name:      "get_weather",
							Arguments: `{"location": "Paris"}`,
						},
					},
				},
			},
		}

		result, _ := convertMessages(messages)

		require.Len(t, result, 2)
		require.NotNil(t, result[1].Parts[0].FunctionCall)
		require.Equal(t, []byte(_THOUGHT_SIGNATURE_BYPASS), result[1].Parts[0].ThoughtSignature)
	})

	t.Run("unknown role returns nil", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: "unknown", Content: providers.ContentFromString("Hello")},
		}

		result, _ := convertMessages(messages)
		require.Empty(t, result)
	})
}

func TestGemini_ConvertFinishReason(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    genai.FinishReason
		expected string
	}{
		{
			name:     "STOP",
			input:    genai.FinishReasonStop,
			expected: providers.FINISH_REASON_STOP,
		},
		{
			name:     "MAX_TOKENS",
			input:    genai.FinishReasonMaxTokens,
			expected: providers.FINISH_REASON_LENGTH,
		},
		{
			name:     "SAFETY",
			input:    genai.FinishReasonSafety,
			expected: providers.FINISH_REASON_CONTENT_FILTER,
		},
		{
			name:     "RECITATION",
			input:    genai.FinishReasonRecitation,
			expected: providers.FINISH_REASON_STOP,
		},
		{
			name:     "BLOCKLIST",
			input:    genai.FinishReasonBlocklist,
			expected: providers.FINISH_REASON_CONTENT_FILTER,
		},
		{
			name:     "PROHIBITED_CONTENT",
			input:    genai.FinishReasonProhibitedContent,
			expected: providers.FINISH_REASON_CONTENT_FILTER,
		},
		{
			name:     "unknown",
			input:    "UNKNOWN",
			expected: providers.FINISH_REASON_STOP,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result := convertFinishReason(tc.input)
			require.Equal(t, tc.expected, result)
		})
	}
}

func TestGemini_ConvertTools(t *testing.T) {
	t.Parallel()

	tools := []providers.ToolInfo{
		{
			Type: "function",
			Function: providers.Function{
				Name:        "get_weather",
				Description: "Get the weather for a location.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{
							"type":        "string",
							"description": "The city name",
						},
					},
					"required": []string{"location"},
				},
			},
		},
	}

	result := convertTools(tools)

	require.Len(t, result, 1)
	require.Len(t, result[0].FunctionDeclarations, 1)
	require.Equal(t, "get_weather", result[0].FunctionDeclarations[0].Name)
	require.Equal(t, "Get the weather for a location.", result[0].FunctionDeclarations[0].Description)
	require.NotNil(t, result[0].FunctionDeclarations[0].ParametersJsonSchema)
}

func TestGemini_ConvertToolChoice(t *testing.T) {
	t.Parallel()

	t.Run("auto string", func(t *testing.T) {
		t.Parallel()

		result := convertToolChoice("auto")
		require.NotNil(t, result)
		require.Equal(t, genai.FunctionCallingConfigModeAuto, result.FunctionCallingConfig.Mode)
	})

	t.Run("none string", func(t *testing.T) {
		t.Parallel()

		result := convertToolChoice("none")
		require.NotNil(t, result)
		require.Equal(t, genai.FunctionCallingConfigModeNone, result.FunctionCallingConfig.Mode)
	})

	t.Run("required string", func(t *testing.T) {
		t.Parallel()

		result := convertToolChoice("required")
		require.NotNil(t, result)
		require.Equal(t, genai.FunctionCallingConfigModeAny, result.FunctionCallingConfig.Mode)
	})

	t.Run("specific function", func(t *testing.T) {
		t.Parallel()

		result := convertToolChoice(providers.ToolChoice{
			Type:     "function",
			Function: &providers.ToolChoiceFunction{Name: "get_weather"},
		})
		require.NotNil(t, result)
		require.Equal(t, genai.FunctionCallingConfigModeAny, result.FunctionCallingConfig.Mode)
		require.Contains(t, result.FunctionCallingConfig.AllowedFunctionNames, "get_weather")
	})

	t.Run("unknown returns nil", func(t *testing.T) {
		t.Parallel()

		result := convertToolChoice("unknown_value")
		require.Nil(t, result)
	})
}

func TestGemini_ConvertError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		err          error
		wantSentinel error
	}{
		{
			name:         "nil error returns nil",
			err:          nil,
			wantSentinel: nil,
		},
		{
			name:         "non-API error becomes ProviderError",
			err:          stderrors.New("network timeout"),
			wantSentinel: errors.ErrProvider,
		},
		{
			name:         "401 status becomes AuthenticationError",
			err:          &genai.APIError{Code: 401, Message: "unauthorized"},
			wantSentinel: errors.ErrAuthentication,
		},
		{
			name:         "403 status becomes AuthenticationError",
			err:          &genai.APIError{Code: 403, Message: "forbidden"},
			wantSentinel: errors.ErrAuthentication,
		},
		{
			name:         "404 status becomes ModelNotFoundError",
			err:          &genai.APIError{Code: 404, Message: "not found"},
			wantSentinel: errors.ErrModelNotFound,
		},
		{
			name:         "429 status becomes RateLimitError",
			err:          &genai.APIError{Code: 429, Message: "rate limited"},
			wantSentinel: errors.ErrRateLimit,
		},
		{
			name:         "400 status becomes InvalidRequestError",
			err:          &genai.APIError{Code: 400, Message: "bad request"},
			wantSentinel: errors.ErrInvalidRequest,
		},
		{
			name:         "400 with context message becomes ContextLengthError",
			err:          &genai.APIError{Code: 400, Message: "context length exceeded"},
			wantSentinel: errors.ErrContextLength,
		},
		{
			name:         "400 with token message becomes ContextLengthError",
			err:          &genai.APIError{Code: 400, Message: "too many tokens in request"},
			wantSentinel: errors.ErrContextLength,
		},
		{
			name:         "400 with safety message becomes ContentFilterError",
			err:          &genai.APIError{Code: 400, Message: "blocked by safety filters"},
			wantSentinel: errors.ErrContentFilter,
		},
		{
			name:         "500 status becomes ProviderError",
			err:          &genai.APIError{Code: 500, Message: "internal error"},
			wantSentinel: errors.ErrProvider,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			p := &Provider{}
			result := p.ConvertError(tc.err)

			if tc.wantSentinel == nil {
				require.Nil(t, result)
				return
			}

			require.NotNil(t, result)
			require.True(
				t,
				stderrors.Is(result, tc.wantSentinel),
				"expected error to match %v, got %v",
				tc.wantSentinel,
				result,
			)
			require.Contains(t, result.Error(), "["+_PROVIDER_NAME+"]")
		})
	}
}

func TestGemini_ThinkingBudget(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		effort   providers.ReasoningEffort
		expected int32
		ok       bool
	}{
		{
			name:     "low effort",
			effort:   providers.REASONING_EFFORT_LOW,
			expected: _THINKING_BUDGET_LOW,
			ok:       true,
		},
		{
			name:     "medium effort",
			effort:   providers.REASONING_EFFORT_MEDIUM,
			expected: _THINKING_BUDGET_MEDIUM,
			ok:       true,
		},
		{
			name:     "high effort",
			effort:   providers.REASONING_EFFORT_HIGH,
			expected: _THINKING_BUDGET_HIGH,
			ok:       true,
		},
		{
			name:     "none effort",
			effort:   providers.REASONING_EFFORT_NONE,
			expected: 0,
			ok:       false,
		},
		{
			name:     "invalid effort",
			effort:   "invalid",
			expected: 0,
			ok:       false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			budget, ok := thinkingBudget(tc.effort)
			require.Equal(t, tc.ok, ok)
			require.Equal(t, tc.expected, budget)
		})
	}
}

func TestGemini_ApplyThinking(t *testing.T) {
	t.Parallel()

	t.Run("empty effort does nothing", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		applyThinking(cfg, "")
		require.Nil(t, cfg.ThinkingConfig)
	})

	t.Run("none effort does nothing", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		applyThinking(cfg, providers.REASONING_EFFORT_NONE)
		require.Nil(t, cfg.ThinkingConfig)
	})

	t.Run("low effort sets thinking config", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		applyThinking(cfg, providers.REASONING_EFFORT_LOW)
		require.NotNil(t, cfg.ThinkingConfig)
		require.True(t, cfg.ThinkingConfig.IncludeThoughts)
		require.Equal(t, _THINKING_BUDGET_LOW, *cfg.ThinkingConfig.ThinkingBudget)
	})

	t.Run("high effort sets thinking config", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		applyThinking(cfg, providers.REASONING_EFFORT_HIGH)
		require.NotNil(t, cfg.ThinkingConfig)
		require.True(t, cfg.ThinkingConfig.IncludeThoughts)
		require.Equal(t, _THINKING_BUDGET_HIGH, *cfg.ThinkingConfig.ThinkingBudget)
	})
}

func TestGemini_ConvertImagePart(t *testing.T) {
	t.Parallel()

	t.Run("converts base64 image", func(t *testing.T) {
		t.Parallel()

		img := &providers.ImageURL{URL: "data:image/jpeg;base64,/9j/4AAQSkZJRg=="}
		result := convertImagePart(img)
		require.NotNil(t, result)
		require.NotNil(t, result.InlineData)
		require.Equal(t, "image/jpeg", result.InlineData.MIMEType)
	})

	t.Run("converts URL image", func(t *testing.T) {
		t.Parallel()

		img := &providers.ImageURL{URL: "https://example.com/image.png"}
		result := convertImagePart(img)
		require.NotNil(t, result)
		require.NotNil(t, result.FileData)
		require.Equal(t, "https://example.com/image.png", result.FileData.FileURI)
	})
}

func TestGemini_ConvertEmbeddingInput(t *testing.T) {
	t.Parallel()

	t.Run("string input", func(t *testing.T) {
		t.Parallel()

		result := convertEmbeddingInput("hello world")
		require.NotNil(t, result)
		require.Len(t, result.Parts, 1)
		require.Equal(t, "hello world", result.Parts[0].Text)
	})

	t.Run("string slice input", func(t *testing.T) {
		t.Parallel()

		result := convertEmbeddingInput([]string{"hello", "world"})
		require.NotNil(t, result)
		require.Len(t, result.Parts, 2)
		require.Equal(t, "hello", result.Parts[0].Text)
		require.Equal(t, "world", result.Parts[1].Text)
	})
}

func TestGemini_GenerateID(t *testing.T) {
	t.Parallel()

	t.Run("has correct prefix", func(t *testing.T) {
		t.Parallel()

		id, err := generateID(_ID_PREFIX_COMPLETION)
		require.NoError(t, err)
		require.True(t, strings.HasPrefix(id, _ID_PREFIX_COMPLETION))
	})

	t.Run("generates unique IDs", func(t *testing.T) {
		t.Parallel()

		id1, err := generateID(_ID_PREFIX_TOOL_CALL)
		require.NoError(t, err)
		id2, err := generateID(_ID_PREFIX_TOOL_CALL)
		require.NoError(t, err)
		require.NotEqual(t, id1, id2)
	})

	t.Run("has expected length", func(t *testing.T) {
		t.Parallel()

		id, err := generateID("test-")
		require.NoError(t, err)
		// prefix (5) + 24 hex chars (12 bytes) = 29.
		require.Len(t, id, 29)
	})
}

func TestGemini_NewStreamState(t *testing.T) {
	t.Parallel()

	state, err := newStreamState("gemini-1.5-flash")
	require.NoError(t, err)
	require.NotNil(t, state)
	require.Equal(t, "gemini-1.5-flash", state.model)
	require.True(t, strings.HasPrefix(state.messageID, _ID_PREFIX_COMPLETION))
	require.Nil(t, state.toolCalls)
	require.Nil(t, state.usage)
}

func TestGemini_SetProviderExtra(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		initial  map[string]providers.ProviderData
		provider string
		key      string
		value    any
		expected map[string]providers.ProviderData
	}{
		{
			name:     "nil Extra initialises both maps",
			initial:  nil,
			provider: _PROVIDER_NAME,
			key:      "thought_signature",
			value:    "abc123",
			expected: map[string]providers.ProviderData{
				_PROVIDER_NAME: {"thought_signature": "abc123"},
			},
		},
		{
			name:     "nil provider map initialises inner map",
			initial:  map[string]providers.ProviderData{},
			provider: _PROVIDER_NAME,
			key:      "thought_signature",
			value:    "abc123",
			expected: map[string]providers.ProviderData{
				_PROVIDER_NAME: {"thought_signature": "abc123"},
			},
		},
		{
			name: "preserves existing provider keys",
			initial: map[string]providers.ProviderData{
				_PROVIDER_NAME: {"existing_key": "existing_value"},
			},
			provider: _PROVIDER_NAME,
			key:      "thought_signature",
			value:    "abc123",
			expected: map[string]providers.ProviderData{
				_PROVIDER_NAME: {
					"existing_key":      "existing_value",
					"thought_signature": "abc123",
				},
			},
		},
		{
			name: "preserves other providers",
			initial: map[string]providers.ProviderData{
				"other": {"key": "value"},
			},
			provider: _PROVIDER_NAME,
			key:      "thought_signature",
			value:    "abc123",
			expected: map[string]providers.ProviderData{
				"other":        {"key": "value"},
				_PROVIDER_NAME: {"thought_signature": "abc123"},
			},
		},
		{
			name: "overwrites existing key",
			initial: map[string]providers.ProviderData{
				_PROVIDER_NAME: {"thought_signature": "old"},
			},
			provider: _PROVIDER_NAME,
			key:      "thought_signature",
			value:    "new",
			expected: map[string]providers.ProviderData{
				_PROVIDER_NAME: {"thought_signature": "new"},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			toolCall := providers.ToolCall{Extra: tc.initial}
			setProviderExtra(&toolCall, tc.provider, tc.key, tc.value)
			require.Equal(t, tc.expected, toolCall.Extra)
		})
	}
}

func TestGemini_ThoughtSignatureFromExtra(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		extra    map[string]providers.ProviderData
		expected []byte
	}{
		{
			name:     "nil extra returns nil",
			extra:    nil,
			expected: nil,
		},
		{
			name:     "missing provider returns nil",
			extra:    map[string]providers.ProviderData{"other": {"key": "value"}},
			expected: nil,
		},
		{
			name: "missing key returns nil",
			extra: map[string]providers.ProviderData{
				_PROVIDER_NAME: {"other_key": "value"},
			},
			expected: nil,
		},
		{
			name: "wrong type returns nil",
			extra: map[string]providers.ProviderData{
				_PROVIDER_NAME: {_EXTRA_KEY_THOUGHT_SIGNATURE: 12345},
			},
			expected: nil,
		},
		{
			name: "invalid base64 returns nil",
			extra: map[string]providers.ProviderData{
				_PROVIDER_NAME: {_EXTRA_KEY_THOUGHT_SIGNATURE: "not-valid-base64!!!"},
			},
			expected: nil,
		},
		{
			name: "valid signature decodes correctly",
			extra: map[string]providers.ProviderData{
				_PROVIDER_NAME: {
					_EXTRA_KEY_THOUGHT_SIGNATURE: base64.StdEncoding.EncodeToString([]byte("test-sig")),
				},
			},
			expected: []byte("test-sig"),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result := thoughtSignatureFromExtra(tc.extra)
			require.Equal(t, tc.expected, result)
		})
	}
}

func TestGemini_ThoughtSignatureRoundTrip(t *testing.T) {
	t.Parallel()

	// Simulate an API response with a ThoughtSignature on a function call.
	originalSig := []byte("opaque-signature-from-gemini-api-xyz123")

	resp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						Name: "search",
						Args: map[string]any{"query": "test"},
					},
					ThoughtSignature: originalSig,
				}},
			},
			FinishReason: genai.FinishReasonStop,
		}},
	}

	// Capture via non-streaming path.
	result, err := convertResponse(resp, "gemini-2.5-pro")
	require.NoError(t, err)
	require.Len(t, result.Choices[0].Message.ToolCalls, 1)

	capturedTC := result.Choices[0].Message.ToolCalls[0]
	require.NotNil(t, capturedTC.Extra)

	// Build a message with the captured tool call (as a caller would).
	assistantMsg := providers.Message{
		Role:      providers.ROLE_ASSISTANT,
		Content:   providers.ContentFromString(""),
		ToolCalls: []providers.ToolCall{capturedTC},
	}

	// Replay via convertAssistantMessage.
	content := convertAssistantMessage(assistantMsg)
	require.NotNil(t, content)
	require.Len(t, content.Parts, 1)

	// Verify the signature round-tripped identically.
	require.Equal(t, originalSig, content.Parts[0].ThoughtSignature)
}

func TestGemini_ThoughtSignatureWireFormat(t *testing.T) {
	t.Parallel()

	t.Run("bypass value is base64-encoded by json.Marshal", func(t *testing.T) {
		t.Parallel()

		// Build a message with no Extra — should get the bypass.
		msg := providers.Message{
			Role: providers.ROLE_ASSISTANT,
			ToolCalls: []providers.ToolCall{{
				ID:   "call_1",
				Type: "function",
				Function: providers.FunctionCall{
					Name:      "search",
					Arguments: `{"q":"test"}`,
				},
			}},
		}

		content := convertAssistantMessage(msg)
		require.Len(t, content.Parts, 1)

		// Marshal the Part as the SDK would before sending.
		raw, err := json.Marshal(content.Parts[0])
		require.NoError(t, err)

		wireJSON := string(raw)

		// The literal bypass must NOT appear — json.Marshal base64-encodes []byte.
		require.NotContains(t, wireJSON, _THOUGHT_SIGNATURE_BYPASS)

		// The base64-encoded form must appear instead.
		encoded := base64.StdEncoding.EncodeToString([]byte(_THOUGHT_SIGNATURE_BYPASS))
		require.Contains(t, wireJSON, encoded)
	})

	t.Run("real signature is base64-encoded by json.Marshal", func(t *testing.T) {
		t.Parallel()

		realSig := []byte("opaque-gemini-signature-abc123")
		storedB64 := base64.StdEncoding.EncodeToString(realSig)

		msg := providers.Message{
			Role: providers.ROLE_ASSISTANT,
			ToolCalls: []providers.ToolCall{{
				ID:   "call_1",
				Type: "function",
				Function: providers.FunctionCall{
					Name:      "search",
					Arguments: `{"q":"test"}`,
				},
				Extra: map[string]providers.ProviderData{
					_PROVIDER_NAME: {_EXTRA_KEY_THOUGHT_SIGNATURE: storedB64},
				},
			}},
		}

		content := convertAssistantMessage(msg)
		require.Len(t, content.Parts, 1)

		// The Part should have the raw bytes.
		require.Equal(t, realSig, content.Parts[0].ThoughtSignature)

		// When marshaled, json.Marshal base64-encodes the raw bytes — which
		// produces a double-encoded value on the wire. This is the expected
		// (if unfortunate) behaviour until the upstream SDK changes
		// ThoughtSignature from []byte to string.
		raw, err := json.Marshal(content.Parts[0])
		require.NoError(t, err)

		wireJSON := string(raw)
		doubleEncoded := base64.StdEncoding.EncodeToString(realSig)
		require.Contains(t, wireJSON, doubleEncoded)
	})
}

func TestGemini_StreamStateProcessResponse(t *testing.T) {
	t.Parallel()

	t.Run("processes text content", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{Text: "Hello "}},
				},
			}},
		}

		chunks, err := state.processResponse(resp)
		require.NoError(t, err)
		require.Len(t, chunks, 1)
		require.Equal(t, "Hello ", chunks[0].Choices[0].Delta.Content)
		require.Equal(t, "Hello ", state.content.String())
	})

	t.Run("processes thinking content", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{Text: "Let me think...", Thought: true}},
				},
			}},
		}

		chunks, err := state.processResponse(resp)
		require.NoError(t, err)
		require.Len(t, chunks, 1)
		require.NotNil(t, chunks[0].Choices[0].Delta.Reasoning)
		require.Equal(t, "Let me think...", chunks[0].Choices[0].Delta.Reasoning.Content)
	})

	t.Run("processes function call", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{
							Name: "get_weather",
							Args: map[string]any{"location": "Paris"},
						},
					}},
				},
			}},
		}

		chunks, err := state.processResponse(resp)
		require.NoError(t, err)
		require.Len(t, chunks, 1)
		require.Len(t, chunks[0].Choices[0].Delta.ToolCalls, 1)
		require.Equal(t, "get_weather", chunks[0].Choices[0].Delta.ToolCalls[0].Function.Name)
		require.Contains(t, chunks[0].Choices[0].Delta.ToolCalls[0].Function.Arguments, "Paris")
	})

	t.Run("tracks usage metadata", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		resp := &genai.GenerateContentResponse{
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:        10,
				CandidatesTokenCount:    5,
				ThoughtsTokenCount:      3,
				CachedContentTokenCount: 4,
				TotalTokenCount:         18,
				PromptTokensDetails: []*genai.ModalityTokenCount{
					{Modality: genai.MediaModalityAudio, TokenCount: 2},
				},
				CandidatesTokensDetails: []*genai.ModalityTokenCount{
					{Modality: genai.MediaModalityAudio, TokenCount: 1},
				},
			},
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{Text: "Hi"}},
				},
			}},
		}

		_, err = state.processResponse(resp)
		require.NoError(t, err)
		require.NotNil(t, state.usage)
		require.Equal(t, 10, state.usage.PromptTokens)
		require.Equal(t, 5, state.usage.CompletionTokens)
		require.Equal(t, 18, state.usage.TotalTokens)
		require.Equal(t, 2, *state.usage.PromptTokensDetails.AudioTokens)
		require.Equal(t, 4, *state.usage.PromptTokensDetails.CachedTokens)
		require.Equal(t, 1, *state.usage.CompletionTokenDetails.AudioTokens)
		require.Equal(t, 3, *state.usage.CompletionTokenDetails.ReasoningTokens)
	})

	t.Run("captures thought signature on function call", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{
							Name: "get_weather",
							Args: map[string]any{"location": "Paris"},
						},
						ThoughtSignature: []byte("test-signature-bytes"),
					}},
				},
			}},
		}

		chunks, err := state.processResponse(resp)
		require.NoError(t, err)
		require.Len(t, chunks, 1)

		tc := chunks[0].Choices[0].Delta.ToolCalls[0]
		require.NotNil(t, tc.Extra)
		geminiData, ok := tc.Extra[_PROVIDER_NAME]
		require.True(t, ok, "expected google provider data in Extra")

		sig, ok := geminiData[_EXTRA_KEY_THOUGHT_SIGNATURE].(string)
		require.True(t, ok, "expected thought_signature to be a string")

		// Value should be base64-encoded.
		decoded, err := base64.StdEncoding.DecodeString(sig)
		require.NoError(t, err)
		require.Equal(t, []byte("test-signature-bytes"), decoded)
	})

	t.Run("no thought signature leaves Extra nil", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{
							Name: "get_weather",
							Args: map[string]any{"location": "Paris"},
						},
					}},
				},
			}},
		}

		chunks, err := state.processResponse(resp)
		require.NoError(t, err)
		require.Len(t, chunks, 1)

		tc := chunks[0].Choices[0].Delta.ToolCalls[0]
		require.Nil(t, tc.Extra)
	})

	t.Run("returns empty slice for empty candidates", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		resp := &genai.GenerateContentResponse{}

		chunks, err := state.processResponse(resp)
		require.NoError(t, err)
		require.Empty(t, chunks)
	})
}

func TestGemini_StreamStateFinalChunk(t *testing.T) {
	t.Parallel()

	t.Run("defaults to stop finish reason", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		state.finishReason = genai.FinishReasonStop

		chunk := state.finalChunk()
		require.Equal(t, providers.FINISH_REASON_STOP, chunk.Choices[0].FinishReason)
	})

	t.Run("uses tool_calls when tool calls present", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		state.finishReason = genai.FinishReasonStop
		state.toolCalls = []providers.ToolCall{
			{ID: "call_1", Type: "function", Function: providers.FunctionCall{Name: "get_weather"}},
		}

		chunk := state.finalChunk()
		require.Equal(t, providers.FINISH_REASON_TOOL_CALLS, chunk.Choices[0].FinishReason)
	})

	t.Run("uses max_tokens finish reason", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		state.finishReason = genai.FinishReasonMaxTokens

		chunk := state.finalChunk()
		require.Equal(t, providers.FINISH_REASON_LENGTH, chunk.Choices[0].FinishReason)
	})

	t.Run("includes usage", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		state.usage = &providers.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}

		chunk := state.finalChunk()
		require.NotNil(t, chunk.Usage)
		require.Equal(t, 15, chunk.Usage.TotalTokens)
	})
}

func TestGemini_ConvertParams(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithAPIKey("test-key"))
	require.NoError(t, err)

	t.Run("json_object sets mime type", func(t *testing.T) {
		t.Parallel()

		_, cfg := provider.convertParams(providers.CompletionParams{
			Model:    "gemini-2.0-flash",
			Messages: []providers.Message{{Role: providers.ROLE_USER, Content: providers.ContentFromString("hi")}},
			ResponseFormat: &providers.ResponseFormat{
				Type: _RESPONSE_FORMAT_JSON,
			},
		})

		require.Equal(t, _RESPONSE_MIME_TYPE_JSON, cfg.ResponseMIMEType)
		require.Nil(t, cfg.ResponseJsonSchema)
	})

	t.Run("json_schema sets mime type and schema", func(t *testing.T) {
		t.Parallel()

		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name":       map[string]any{"type": "string"},
				"population": map[string]any{"type": "integer"},
			},
			"required": []string{"name", "population"},
		}

		_, cfg := provider.convertParams(providers.CompletionParams{
			Model:    "gemini-2.0-flash",
			Messages: []providers.Message{{Role: providers.ROLE_USER, Content: providers.ContentFromString("hi")}},
			ResponseFormat: &providers.ResponseFormat{
				Type: _RESPONSE_FORMAT_JSON_SCHEMA,
				JSONSchema: &providers.JSONSchema{
					Name:   "city_info",
					Schema: schema,
				},
			},
		})

		require.Equal(t, _RESPONSE_MIME_TYPE_JSON, cfg.ResponseMIMEType)
		require.Equal(t, schema, cfg.ResponseJsonSchema)
	})

	t.Run("nil response format leaves config unchanged", func(t *testing.T) {
		t.Parallel()

		_, cfg := provider.convertParams(providers.CompletionParams{
			Model:    "gemini-2.0-flash",
			Messages: []providers.Message{{Role: providers.ROLE_USER, Content: providers.ContentFromString("hi")}},
		})

		require.Empty(t, cfg.ResponseMIMEType)
		require.Nil(t, cfg.ResponseJsonSchema)
	})
}

func TestGemini_ConvertResponse(t *testing.T) {
	t.Parallel()

	t.Run("converts text response", func(t *testing.T) {
		t.Parallel()

		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{Text: "Hello World"}},
				},
				FinishReason: genai.FinishReasonStop,
			}},
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:        10,
				CandidatesTokenCount:    5,
				ThoughtsTokenCount:      3,
				CachedContentTokenCount: 4,
				TotalTokenCount:         18,
				PromptTokensDetails: []*genai.ModalityTokenCount{
					{Modality: genai.MediaModalityAudio, TokenCount: 2},
				},
				CandidatesTokensDetails: []*genai.ModalityTokenCount{
					{Modality: genai.MediaModalityAudio, TokenCount: 1},
				},
			},
		}

		result, err := convertResponse(resp, "gemini-1.5-flash")
		require.NoError(t, err)
		require.Equal(t, _OBJECT_CHAT_COMPLETION, result.Object)
		require.Equal(t, "gemini-1.5-flash", result.Model)
		require.Len(t, result.Choices, 1)
		require.Equal(t, "Hello World", result.Choices[0].Message.ContentString())
		require.Equal(t, providers.ROLE_ASSISTANT, result.Choices[0].Message.Role)
		require.Equal(t, providers.FINISH_REASON_STOP, result.Choices[0].FinishReason)
		require.NotNil(t, result.Usage)
		require.Equal(t, 10, result.Usage.PromptTokens)
		require.Equal(t, 5, result.Usage.CompletionTokens)
		require.Equal(t, 18, result.Usage.TotalTokens)
		require.Equal(t, 2, *result.Usage.PromptTokensDetails.AudioTokens)
		require.Equal(t, 4, *result.Usage.PromptTokensDetails.CachedTokens)
		require.Equal(t, 1, *result.Usage.CompletionTokenDetails.AudioTokens)
		require.Equal(t, 3, *result.Usage.CompletionTokenDetails.ReasoningTokens)
	})

	t.Run("converts function call response", func(t *testing.T) {
		t.Parallel()

		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{
							Name: "get_weather",
							Args: map[string]any{"location": "Paris"},
						},
					}},
				},
				FinishReason: genai.FinishReasonStop,
			}},
		}

		result, err := convertResponse(resp, "gemini-1.5-flash")
		require.NoError(t, err)
		require.Len(t, result.Choices[0].Message.ToolCalls, 1)
		require.Equal(t, "get_weather", result.Choices[0].Message.ToolCalls[0].Function.Name)
		require.Equal(t, providers.FINISH_REASON_TOOL_CALLS, result.Choices[0].FinishReason)
	})

	t.Run("captures thought signature on function call", func(t *testing.T) {
		t.Parallel()

		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{
							Name: "get_weather",
							Args: map[string]any{"location": "Paris"},
						},
						ThoughtSignature: []byte("non-stream-sig"),
					}},
				},
				FinishReason: genai.FinishReasonStop,
			}},
		}

		result, err := convertResponse(resp, "gemini-2.5-pro")
		require.NoError(t, err)
		require.Len(t, result.Choices[0].Message.ToolCalls, 1)

		tc := result.Choices[0].Message.ToolCalls[0]
		require.NotNil(t, tc.Extra)
		geminiData := tc.Extra[_PROVIDER_NAME]
		sig, ok := geminiData[_EXTRA_KEY_THOUGHT_SIGNATURE].(string)
		require.True(t, ok)

		decoded, err := base64.StdEncoding.DecodeString(sig)
		require.NoError(t, err)
		require.Equal(t, []byte("non-stream-sig"), decoded)
	})

	t.Run("converts thinking response", func(t *testing.T) {
		t.Parallel()

		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{Text: "Let me think...", Thought: true},
						{Text: "Hello!"},
					},
				},
				FinishReason: genai.FinishReasonStop,
			}},
		}

		result, err := convertResponse(resp, "gemini-2.0-flash")
		require.NoError(t, err)
		require.Equal(t, "Hello!", result.Choices[0].Message.ContentString())
		require.NotNil(t, result.Choices[0].Message.Reasoning)
		require.Equal(t, "Let me think...", result.Choices[0].Message.Reasoning.Content)
	})
}

func TestGemini_ApplyResponseFormat(t *testing.T) {
	t.Parallel()

	t.Run("json_object sets mime type", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		applyResponseFormat(cfg, &providers.ResponseFormat{Type: "json_object"})
		require.Equal(t, "application/json", cfg.ResponseMIMEType)
	})

	t.Run("text does not set mime type", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		applyResponseFormat(cfg, &providers.ResponseFormat{Type: "text"})
		require.Empty(t, cfg.ResponseMIMEType)
	})

	t.Run("json_schema sets mime type and schema", func(t *testing.T) {
		t.Parallel()

		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
			},
		}
		cfg := &genai.GenerateContentConfig{}
		applyResponseFormat(cfg, &providers.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &providers.JSONSchema{
				Name:   "test_schema",
				Schema: schema,
			},
		})
		require.Equal(t, "application/json", cfg.ResponseMIMEType)
		require.Equal(t, schema, cfg.ResponseJsonSchema)
	})

	t.Run("json_schema with nil JSONSchema does not set mime type", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		applyResponseFormat(cfg, &providers.ResponseFormat{Type: "json_schema"})
		require.Empty(t, cfg.ResponseMIMEType)
		require.Nil(t, cfg.ResponseJsonSchema)
	})

	t.Run("json_schema with nil Schema does not set mime type", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		applyResponseFormat(cfg, &providers.ResponseFormat{
			Type:       "json_schema",
			JSONSchema: &providers.JSONSchema{Name: "test"},
		})
		require.Empty(t, cfg.ResponseMIMEType)
		require.Nil(t, cfg.ResponseJsonSchema)
	})
}

// Integration tests - only run if API key is available.

func TestGemini_IntegrationCompletion(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(_PROVIDER_NAME) {
		t.Skip("GEMINI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel(_PROVIDER_NAME),
		Messages: testutil.SimpleMessages(),
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Equal(t, _OBJECT_CHAT_COMPLETION, resp.Object)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.Content)
	require.Equal(t, providers.ROLE_ASSISTANT, resp.Choices[0].Message.Role)
	require.NotNil(t, resp.Usage)
	require.Greater(t, resp.Usage.TotalTokens, 0)
}

func TestGemini_IntegrationCompletionWithSystemMessage(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(_PROVIDER_NAME) {
		t.Skip("GEMINI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel(_PROVIDER_NAME),
		Messages: testutil.MessagesWithSystem(),
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.Content)
}

func TestGemini_IntegrationCompletionStream(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(_PROVIDER_NAME) {
		t.Skip("GEMINI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel(_PROVIDER_NAME),
		Messages: testutil.SimpleMessages(),
		Stream:   true,
	}

	chunks, errs := provider.CompletionStream(ctx, params)

	var content strings.Builder
	chunkCount := 0

	for chunk := range chunks {
		chunkCount++
		require.Equal(t, _OBJECT_CHAT_COMPLETION_CHUNK, chunk.Object)
		if len(chunk.Choices) > 0 {
			content.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	require.NoError(t, <-errs)
	require.Greater(t, chunkCount, 0)
	require.NotEmpty(t, content.String())
}

func TestGemini_CompletionStreamContextCancellation(t *testing.T) {
	requestStarted := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		close(requestStarted)
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)

	client, err := genai.NewClient(t.Context(), &genai.ClientConfig{
		APIKey:  "test-key",
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL: srv.URL,
		},
	})
	require.NoError(t, err)
	provider := &Provider{client: client}
	params := providers.CompletionParams{
		Model:    "test-model",
		Messages: []providers.Message{{Role: providers.ROLE_USER, Content: providers.ContentFromString("Hello")}},
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		select {
		case <-requestStarted:
			cancel()
		case <-time.After(time.Second):
			cancel()
		}
	}()

	chunks, errs := provider.CompletionStream(ctx, params)
	for range chunks {
	}
	require.ErrorIs(t, <-errs, context.Canceled)
}

func TestGemini_CompletionStreamCancellationClosesRequest(t *testing.T) {
	requestDone := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write(
			[]byte(
				`data: {"candidates":[{"content":{"parts":[{"text":"hello"}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}

`,
			),
		)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-r.Context().Done()
		close(requestDone)
	}))
	t.Cleanup(srv.Close)

	client, err := genai.NewClient(t.Context(), &genai.ClientConfig{
		APIKey:  "test-key",
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL: srv.URL,
		},
	})
	require.NoError(t, err)
	provider := &Provider{client: client}
	params := providers.CompletionParams{
		Model:    "test-model",
		Messages: []providers.Message{{Role: providers.ROLE_USER, Content: providers.ContentFromString("Hello")}},
	}

	ctx, cancel := context.WithCancel(t.Context())
	chunks, errs := provider.CompletionStream(ctx, params)

	// Receive the first chunk, then cancel the stream.
	first, ok := <-chunks
	require.True(t, ok)
	require.Equal(t, _OBJECT_CHAT_COMPLETION_CHUNK, first.Object)
	cancel()

	select {
	case <-requestDone:
	case <-time.After(time.Second):
		t.Fatal("request remained active after cancellation")
	}

	require.ErrorIs(t, <-errs, context.Canceled)
}

func TestGemini_IntegrationCompletionConversation(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(_PROVIDER_NAME) {
		t.Skip("GEMINI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel(_PROVIDER_NAME),
		Messages: testutil.ConversationMessages(),
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)

	contentStr := resp.Choices[0].Message.ContentString()
	require.Contains(t, strings.ToLower(contentStr), "alice")
}

func TestGemini_IntegrationEmbedding(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(_PROVIDER_NAME) {
		t.Skip("GEMINI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.EmbeddingParams{
		Model: testutil.EmbeddingModel(_PROVIDER_NAME),
		Input: "Hello world",
	}

	resp, err := provider.Embedding(ctx, params)
	require.NoError(t, err)

	require.Equal(t, _OBJECT_LIST, resp.Object)
	require.NotEmpty(t, resp.Data)
	require.NotEmpty(t, resp.Data[0].Embedding)
	require.Equal(t, _OBJECT_EMBEDDING, resp.Data[0].Object)
}

func TestGemini_IntegrationListModels(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(_PROVIDER_NAME) {
		t.Skip("GEMINI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	resp, err := provider.ListModels(ctx)
	require.NoError(t, err)

	require.Equal(t, _OBJECT_LIST, resp.Object)
	require.NotEmpty(t, resp.Data)

	// Verify model structure.
	for _, model := range resp.Data {
		require.NotEmpty(t, model.ID)
		require.Equal(t, _OBJECT_MODEL, model.Object)
		require.Equal(t, "google", model.OwnedBy)
	}
}
