package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	openaisdk "github.com/openai/openai-go"
	"github.com/stretchr/testify/require"

	"github.com/humbornjo/llm/config"
	"github.com/humbornjo/llm/errors"
	"github.com/humbornjo/llm/providers"
)

func TestOpenAI_ConvertUsage(t *testing.T) {
	t.Parallel()

	var usage openaisdk.CompletionUsage
	require.NoError(t, json.Unmarshal([]byte(`{
		"prompt_tokens": 12,
		"completion_tokens": 8,
		"total_tokens": 20,
		"prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 4},
		"completion_tokens_details": {
			"accepted_prediction_tokens": 2,
			"audio_tokens": 1,
			"reasoning_tokens": 3,
			"rejected_prediction_tokens": 0
		}
	}`), &usage))

	result := convertUsage(usage)
	require.Equal(t, 12, result.PromptTokens)
	require.Equal(t, 8, result.CompletionTokens)
	require.Equal(t, 20, result.TotalTokens)
	require.Equal(t, 0, *result.PromptTokensDetails.AudioTokens)
	require.Equal(t, 4, *result.PromptTokensDetails.CachedTokens)
	require.Nil(t, result.PromptTokensDetails.CacheWriteTokens)
	require.Equal(t, 2, *result.CompletionTokenDetails.AcceptedPredictionTokens)
	require.Equal(t, 1, *result.CompletionTokenDetails.AudioTokens)
	require.Equal(t, 3, *result.CompletionTokenDetails.ReasoningTokens)
	require.Equal(t, 0, *result.CompletionTokenDetails.RejectedPredictionTokens)
}

func TestOpenAI_NewCompatible(t *testing.T) {
	// Note: Not using t.Parallel() here because child test uses t.Setenv.

	t.Run("creates provider with valid config", func(t *testing.T) {
		t.Parallel()

		baseCfg := CompatibleConfig{
			Name:           "test-provider",
			DefaultBaseURL: "http://localhost:8080/v1",
			DefaultAPIKey:  "test-key",
			RequireAPIKey:  false,
			Capabilities: providers.Capabilities{
				Completion: true,
			},
		}

		provider, err := NewCompatible(baseCfg)
		require.NoError(t, err)
		require.NotNil(t, provider)
		require.Equal(t, "test-provider", provider.Name())
	})

	t.Run("returns error when name is missing", func(t *testing.T) {
		t.Parallel()

		baseCfg := CompatibleConfig{
			DefaultBaseURL: "http://localhost:8080/v1",
		}

		provider, err := NewCompatible(baseCfg)
		require.Error(t, err)
		require.Nil(t, provider)
		require.Contains(t, err.Error(), "provider name is required")
	})

	t.Run("returns error when API key required but missing", func(t *testing.T) {
		t.Parallel()

		baseCfg := CompatibleConfig{
			Name:          "test-provider",
			APIKeyEnvVar:  "TEST_API_KEY",
			RequireAPIKey: true,
		}

		provider, err := NewCompatible(baseCfg)
		require.Error(t, err)
		require.Nil(t, provider)

		var missingKeyErr *errors.MissingAPIKeyError
		require.ErrorAs(t, err, &missingKeyErr)
	})

	t.Run("uses default API key when not required", func(t *testing.T) {
		t.Parallel()

		baseCfg := CompatibleConfig{
			Name:          "test-provider",
			DefaultAPIKey: "default-key",
			RequireAPIKey: false,
		}

		provider, err := NewCompatible(baseCfg)
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("uses config base URL over default", func(t *testing.T) {
		t.Parallel()

		baseCfg := CompatibleConfig{
			Name:           "test-provider",
			DefaultBaseURL: "http://default:8080/v1",
			DefaultAPIKey:  "test-key",
		}

		provider, err := NewCompatible(baseCfg, config.WithBaseURL("http://custom:9090/v1"))
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("uses environment variable for base URL", func(t *testing.T) {
		t.Setenv("TEST_BASE_URL", "http://env:8080/v1")

		baseCfg := CompatibleConfig{
			Name:           "test-provider",
			BaseURLEnvVar:  "TEST_BASE_URL",
			DefaultBaseURL: "http://default:8080/v1",
			DefaultAPIKey:  "test-key",
		}

		provider, err := NewCompatible(baseCfg)
		require.NoError(t, err)
		require.NotNil(t, provider)
	})
}

func TestOpenAI_NewCompatibleRequireBaseURL(t *testing.T) {
	// Note: Not using t.Parallel() because subtests use t.Setenv.

	const (
		envVar       = "TEST_COMPATIBLE_REQUIRE_BASEURL"
		providerName = "test-provider"
	)

	tests := []struct {
		name           string
		baseURLEnvVar  string
		defaultBaseURL string
		envValue       string
		withBaseURL    string
		requireBaseURL bool
		wantErr        string // empty means no error expected.
	}{
		{
			name:           "errors when required and no env var configured",
			requireBaseURL: true,
			wantErr:        providerName + " base URL is required (set via WithBaseURL option)",
		},
		{
			name:           "errors when required and env var name set but unset",
			baseURLEnvVar:  envVar,
			requireBaseURL: true,
			wantErr: providerName + ` base URL is required (set via WithBaseURL option or "` +
				envVar + `" env var)`,
		},
		{
			name:           "succeeds when required and WithBaseURL is provided",
			requireBaseURL: true,
			withBaseURL:    "http://custom:9090/v1",
		},
		{
			name:           "succeeds when required and env var resolves",
			baseURLEnvVar:  envVar,
			envValue:       "http://env:8080/v1",
			requireBaseURL: true,
		},
		{
			name:           "succeeds when required and DefaultBaseURL is set",
			defaultBaseURL: "http://default:8080/v1",
			requireBaseURL: true,
		},
		{
			name:           "does not error when not required and no URL resolves",
			requireBaseURL: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(envVar, tc.envValue)

			baseCfg := CompatibleConfig{
				Name:           providerName,
				BaseURLEnvVar:  tc.baseURLEnvVar,
				DefaultAPIKey:  "test-key",
				DefaultBaseURL: tc.defaultBaseURL,
				RequireBaseURL: tc.requireBaseURL,
			}

			var opts []config.Option
			if tc.withBaseURL != "" {
				opts = append(opts, config.WithBaseURL(tc.withBaseURL))
			}

			provider, err := NewCompatible(baseCfg, opts...)

			if tc.wantErr != "" {
				require.EqualError(t, err, tc.wantErr)
				require.Nil(t, provider)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, provider)
		})
	}
}

func TestOpenAI_CompatibleProviderCapabilities(t *testing.T) {
	t.Parallel()

	expectedCaps := providers.Capabilities{
		Completion:          true,
		CompletionStreaming: true,
		Embedding:           true,
	}

	baseCfg := CompatibleConfig{
		Name:         "test-provider",
		Capabilities: expectedCaps,
	}

	provider, err := NewCompatible(baseCfg)
	require.NoError(t, err)

	caps := provider.Capabilities()
	require.Equal(t, expectedCaps, caps)
}

func TestOpenAI_ValidateCompletionParams(t *testing.T) {
	t.Parallel()

	t.Run("returns error when model is empty", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Messages: []providers.Message{{Role: providers.ROLE_USER, Content: providers.ContentFromString("Hello")}},
		}

		err := validateCompletionParams(params)
		require.Error(t, err)
		require.Contains(t, err.Error(), "model is required")
	})

	t.Run("returns error when messages is empty", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model:    "gpt-4",
			Messages: []providers.Message{},
		}

		err := validateCompletionParams(params)
		require.Error(t, err)
		require.Contains(t, err.Error(), "at least one message is required")
	})

	t.Run("returns error for unknown message role", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model: "gpt-4",
			Messages: []providers.Message{
				{Role: "unknown_role", Content: providers.ContentFromString("Hello")},
			},
		}

		err := validateCompletionParams(params)
		require.Error(t, err)
		require.Contains(t, err.Error(), "unknown message role")
	})

	t.Run("accepts valid params", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model: "gpt-4",
			Messages: []providers.Message{
				{Role: providers.ROLE_USER, Content: providers.ContentFromString("Hello")},
			},
		}

		err := validateCompletionParams(params)
		require.NoError(t, err)
	})
}

func TestOpenAI_ConvertResponseFormat(t *testing.T) {
	t.Parallel()

	t.Run("handles nil format", func(t *testing.T) {
		t.Parallel()

		result := convertResponseFormat(nil)
		require.NotNil(t, result)
	})

	t.Run("converts json_object format", func(t *testing.T) {
		t.Parallel()

		format := &providers.ResponseFormat{Type: _RESPONSE_FORMAT_JSON_OBJECT}
		result := convertResponseFormat(format)
		require.NotNil(t, result.OfJSONObject)
	})

	t.Run("converts json_schema format", func(t *testing.T) {
		t.Parallel()

		strict := true
		format := &providers.ResponseFormat{
			Type: _RESPONSE_FORMAT_JSON_SCHEMA,
			JSONSchema: &providers.JSONSchema{
				Name:        "test_schema",
				Description: "Test schema",
				Schema:      map[string]any{"type": "object"},
				Strict:      &strict,
			},
		}
		result := convertResponseFormat(format)
		require.NotNil(t, result.OfJSONSchema)
	})

	t.Run("defaults to text format for unknown type", func(t *testing.T) {
		t.Parallel()

		format := &providers.ResponseFormat{Type: "unknown"}
		result := convertResponseFormat(format)
		require.NotNil(t, result.OfText)
	})
}

func TestOpenAI_ConvertAssistantMessage(t *testing.T) {
	t.Parallel()

	t.Run("omits content on a tool-call-only message", func(t *testing.T) {
		t.Parallel()

		converted, err := convertAssistantMessage(providers.Message{
			Role: providers.ROLE_ASSISTANT,
			ToolCalls: []providers.ToolCall{{
				ID:       "call-1",
				Type:     "function",
				Function: providers.FunctionCall{Name: "bash", Arguments: `{"command":"pwd"}`},
			}},
		})
		require.NoError(t, err)
		raw, err := json.Marshal(converted)
		require.NoError(t, err)
		var fields map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(raw, &fields))
		require.NotContains(t, fields, "content")
		require.Contains(t, string(raw), `"tool_calls"`)
	})

	t.Run("omits empty string content beside tool calls", func(t *testing.T) {
		t.Parallel()

		converted, err := convertAssistantMessage(providers.Message{
			Role:    providers.ROLE_ASSISTANT,
			Content: providers.ContentFromString(""),
			ToolCalls: []providers.ToolCall{{
				ID:       "call-1",
				Type:     "function",
				Function: providers.FunctionCall{Name: "bash", Arguments: `{"command":"pwd"}`},
			}},
		})
		require.NoError(t, err)
		raw, err := json.Marshal(converted)
		require.NoError(t, err)
		var fields map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(raw, &fields))
		require.NotContains(t, fields, "content")
	})

	t.Run("keeps text content beside tool calls", func(t *testing.T) {
		t.Parallel()

		converted, err := convertAssistantMessage(providers.Message{
			Role:    providers.ROLE_ASSISTANT,
			Content: providers.ContentFromString("on it"),
			ToolCalls: []providers.ToolCall{{
				ID:       "call-1",
				Type:     "function",
				Function: providers.FunctionCall{Name: "bash", Arguments: `{"command":"pwd"}`},
			}},
		})
		require.NoError(t, err)
		raw, err := json.Marshal(converted)
		require.NoError(t, err)
		require.Contains(t, string(raw), `"content":"on it"`)
	})

	t.Run("flattens text parts without tool calls", func(t *testing.T) {
		t.Parallel()

		converted, err := convertAssistantMessage(providers.Message{
			Role: providers.ROLE_ASSISTANT,
			Content: providers.ContentFromParts(
				&providers.ContentPartText{Text: "Hi there! "},
				&providers.ContentPartText{Text: "How can I help?"},
			),
		})
		require.NoError(t, err)
		raw, err := json.Marshal(converted)
		require.NoError(t, err)
		require.Contains(t, string(raw), `"content":"Hi there! How can I help?"`)
	})

	t.Run("flattens text parts beside tool calls", func(t *testing.T) {
		t.Parallel()

		converted, err := convertAssistantMessage(providers.Message{
			Role:    providers.ROLE_ASSISTANT,
			Content: providers.ContentFromParts(&providers.ContentPartText{Text: "on it"}),
			ToolCalls: []providers.ToolCall{{
				ID:       "call-1",
				Type:     "function",
				Function: providers.FunctionCall{Name: "bash", Arguments: `{"command":"pwd"}`},
			}},
		})
		require.NoError(t, err)
		raw, err := json.Marshal(converted)
		require.NoError(t, err)
		require.Contains(t, string(raw), `"content":"on it"`)
	})

	t.Run("rejects non-text parts", func(t *testing.T) {
		t.Parallel()

		_, err := convertAssistantMessage(providers.Message{
			Role: providers.ROLE_ASSISTANT,
			Content: providers.ContentFromParts(
				&providers.ContentPartText{Text: "look: "},
				&providers.ContentPartImage{ImageURL: &providers.ImageURL{URL: "https://example.com/a.png"}},
			),
		})
		require.Error(t, err)
		require.Contains(t, err.Error(), "only text parts")
	})
}

func TestOpenAI_ConvertUserMessage(t *testing.T) {
	t.Parallel()

	t.Run("converts every part kind", func(t *testing.T) {
		t.Parallel()

		converted, err := convertUserMessage(providers.Message{
			Role: providers.ROLE_USER,
			Content: providers.ContentFromParts(
				&providers.ContentPartText{Text: "describe these"},
				&providers.ContentPartImage{ImageURL: &providers.ImageURL{
					URL:    "https://example.com/a.png",
					Detail: "high",
				}},
				&providers.ContentPartAudio{InputAudio: &providers.InputAudio{Data: "AQI=", Format: "wav"}},
				&providers.ContentPartFile{File: &providers.File{
					FileId:   "file-1",
					FileName: "notes.pdf",
					FileData: "JVBERi0=",
				}},
			),
		})
		require.NoError(t, err)
		raw, err := json.Marshal(converted)
		require.NoError(t, err)
		require.Contains(t, string(raw), `{"text":"describe these","type":"text"}`)
		require.Contains(t, string(raw),
			`{"image_url":{"url":"https://example.com/a.png","detail":"high"},"type":"image_url"}`)
		require.Contains(t, string(raw),
			`{"input_audio":{"data":"AQI=","format":"wav"},"type":"input_audio"}`)
		require.Contains(t, string(raw),
			`{"file":{"file_data":"JVBERi0=","file_id":"file-1","filename":"notes.pdf"},"type":"file"}`)
	})

	t.Run("omits unset file fields", func(t *testing.T) {
		t.Parallel()

		converted, err := convertUserMessage(providers.Message{
			Role: providers.ROLE_USER,
			Content: providers.ContentFromParts(
				&providers.ContentPartFile{File: &providers.File{FileId: "file-1"}},
			),
		})
		require.NoError(t, err)
		raw, err := json.Marshal(converted)
		require.NoError(t, err)
		require.Contains(t, string(raw), `{"file":{"file_id":"file-1"},"type":"file"}`)
	})

	t.Run("rejects parts without their payload", func(t *testing.T) {
		t.Parallel()

		for _, tc := range []struct {
			name    string
			part    providers.IsContentPart
			wantErr string
		}{
			{"image", &providers.ContentPartImage{}, "requires image_url"},
			{"audio", &providers.ContentPartAudio{}, "requires input_audio"},
			{"file", &providers.ContentPartFile{}, "requires file"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()

				_, err := convertUserMessage(providers.Message{
					Role:    providers.ROLE_USER,
					Content: providers.ContentFromParts(tc.part),
				})
				require.Error(t, err)
				require.Contains(t, err.Error(), tc.wantErr)
			})
		}
	})
}

func TestOpenAI_ConvertEmbeddingParams(t *testing.T) {
	t.Parallel()

	t.Run("converts string input", func(t *testing.T) {
		t.Parallel()

		params := providers.EmbeddingParams{
			Model: "text-embedding-3-small",
			Input: "Hello, world!",
		}

		result := convertEmbeddingParams(params)
		require.NotNil(t, result.Input.OfString)
	})

	t.Run("converts string array input", func(t *testing.T) {
		t.Parallel()

		params := providers.EmbeddingParams{
			Model: "text-embedding-3-small",
			Input: []string{"Hello", "World"},
		}

		result := convertEmbeddingParams(params)
		require.NotNil(t, result.Input.OfArrayOfStrings)
	})

	t.Run("handles unknown input type", func(t *testing.T) {
		t.Parallel()

		params := providers.EmbeddingParams{
			Model: "text-embedding-3-small",
			Input: 12345, // Unsupported type.
		}

		result := convertEmbeddingParams(params)
		// Should convert to string representation.
		require.NotNil(t, result.Input.OfString)
	})

	t.Run("includes optional parameters", func(t *testing.T) {
		t.Parallel()

		dims := 256
		params := providers.EmbeddingParams{
			Model:          "text-embedding-3-small",
			Input:          "Hello",
			EncodingFormat: "float",
			Dimensions:     &dims,
			User:           "test-user",
		}

		result := convertEmbeddingParams(params)
		require.Equal(t, int64(256), result.Dimensions.Value)
		require.Equal(t, "test-user", result.User.Value)
	})
}

func TestOpenAI_StreamingContextCancellation(t *testing.T) {
	t.Parallel()

	t.Run("respects context cancellation", func(t *testing.T) {
		t.Parallel()

		baseCfg := CompatibleConfig{
			Name:           "test-provider",
			DefaultBaseURL: "http://localhost:9999/v1", // Non-existent server.
			DefaultAPIKey:  "test-key",
		}

		provider, err := NewCompatible(baseCfg)
		require.NoError(t, err)

		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately.

		params := providers.CompletionParams{
			Model:    "test-model",
			Messages: []providers.Message{{Role: providers.ROLE_USER, Content: providers.ContentFromString("Hello")}},
		}

		chunks, errs := provider.CompletionStream(ctx, params)
		for range chunks {
		}
		require.ErrorIs(t, <-errs, context.Canceled)
	})

	t.Run("preserves deadline errors", func(t *testing.T) {
		t.Parallel()

		provider, err := NewCompatible(CompatibleConfig{
			Name:           "test-provider",
			DefaultBaseURL: "http://localhost:9999/v1",
			DefaultAPIKey:  "test-key",
		})
		require.NoError(t, err)
		ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
		defer cancel()

		chunks, errs := provider.CompletionStream(ctx, providers.CompletionParams{
			Model:    "test-model",
			Messages: []providers.Message{{Role: providers.ROLE_USER, Content: providers.ContentFromString("Hello")}},
		})
		for range chunks {
		}
		require.ErrorIs(t, <-errs, context.DeadlineExceeded)
	})

	// Regression for #85: context cancellation must be yielded explicitly so
	// consumers can distinguish it from a clean end of iteration.
	t.Run("surfaces ctx.Err on cancellation", func(t *testing.T) {
		t.Parallel()

		// Slow upstream: holds the connection open until the test cancels.
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			<-r.Context().Done()
		}))
		t.Cleanup(srv.Close)

		provider, err := NewCompatible(CompatibleConfig{
			Name:           "test-provider",
			DefaultBaseURL: srv.URL + "/v1",
			DefaultAPIKey:  "test-key",
		})
		require.NoError(t, err)

		ctx, cancel := context.WithCancel(context.Background())
		params := providers.CompletionParams{
			Model:    "test-model",
			Messages: []providers.Message{{Role: providers.ROLE_USER, Content: providers.ContentFromString("Hello")}},
		}

		go func() {
			time.Sleep(50 * time.Millisecond)
			cancel()
		}()

		chunks, errs := provider.CompletionStream(ctx, params)
		chunkCount := 0
		for range chunks {
			chunkCount++
		}
		require.ErrorIs(t, <-errs, context.Canceled)
		require.Zero(t, chunkCount)
	})
}

func TestOpenAI_CompletionStreamLifecycle(t *testing.T) {
	params := providers.CompletionParams{
		Model:    "test-model",
		Messages: []providers.Message{{Role: providers.ROLE_USER, Content: providers.ContentFromString("Hello")}},
	}

	t.Run("cancellation closes the request", func(t *testing.T) {
		requestDone := make(chan struct{})
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write(
				[]byte(
					"data: {\"id\":\"chunk\",\"object\":\"chat.completion.chunk\",\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n",
				),
			)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			<-r.Context().Done()
			close(requestDone)
		}))
		t.Cleanup(srv.Close)

		provider, err := NewCompatible(CompatibleConfig{
			Name:           "test-provider",
			DefaultBaseURL: srv.URL + "/v1",
			DefaultAPIKey:  "test-key",
		})
		require.NoError(t, err)

		ctx, cancel := context.WithCancel(t.Context())
		chunks, errs := provider.CompletionStream(ctx, params)

		// Receive the first chunk, then cancel the stream.
		first, ok := <-chunks
		require.True(t, ok)
		require.Equal(t, "chunk", first.ID)
		cancel()

		select {
		case <-requestDone:
		case <-time.After(time.Second):
			t.Fatal("request remained active after cancellation")
		}

		require.ErrorIs(t, <-errs, context.Canceled)
	})

	t.Run("producer blocked on send unblocks on cancellation", func(t *testing.T) {
		requestDone := make(chan struct{})
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write(
				[]byte(
					`data: {"id":"chunk-1","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{"content":"one"}}]}

data: {"id":"chunk-2","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{"content":"two"}}]}

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

		provider, err := NewCompatible(CompatibleConfig{
			Name:           "test-provider",
			DefaultBaseURL: srv.URL + "/v1",
			DefaultAPIKey:  "test-key",
		})
		require.NoError(t, err)

		ctx, cancel := context.WithCancel(t.Context())
		chunks, errs := provider.CompletionStream(ctx, params)

		// Read one chunk, then stop consuming. The producer blocks sending
		// the second chunk until cancellation releases it.
		first, ok := <-chunks
		require.True(t, ok)
		require.Equal(t, "chunk-1", first.ID)
		cancel()

		select {
		case <-requestDone:
		case <-time.After(time.Second):
			t.Fatal("cancellation did not close the blocked stream request")
		}
		require.ErrorIs(t, <-errs, context.Canceled)
	})
}
