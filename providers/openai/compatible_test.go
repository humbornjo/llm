package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/humbornjo/llm/config"
	"github.com/humbornjo/llm/errors"
	"github.com/humbornjo/llm/providers"
)

func TestNewCompatible(t *testing.T) {
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

func TestNewCompatibleRequireBaseURL(t *testing.T) {
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

func TestCompatibleProviderCapabilities(t *testing.T) {
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

func TestValidateCompletionParams(t *testing.T) {
	t.Parallel()

	t.Run("returns error when model is empty", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Messages: []providers.Message{{Role: providers.RoleUser, Content: "Hello"}},
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
				{Role: "unknown_role", Content: "Hello"},
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
				{Role: providers.RoleUser, Content: "Hello"},
			},
		}

		err := validateCompletionParams(params)
		require.NoError(t, err)
	})
}

func TestConvertResponseFormat(t *testing.T) {
	t.Parallel()

	t.Run("handles nil format", func(t *testing.T) {
		t.Parallel()

		result := convertResponseFormat(nil)
		require.NotNil(t, result)
	})

	t.Run("converts json_object format", func(t *testing.T) {
		t.Parallel()

		format := &providers.ResponseFormat{Type: responseFormatJSONObject}
		result := convertResponseFormat(format)
		require.NotNil(t, result.OfJSONObject)
	})

	t.Run("converts json_schema format", func(t *testing.T) {
		t.Parallel()

		strict := true
		format := &providers.ResponseFormat{
			Type: responseFormatJSONSchema,
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

func TestConvertEmbeddingParams(t *testing.T) {
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

func TestStreamingContextCancellation(t *testing.T) {
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
			Messages: []providers.Message{{Role: providers.RoleUser, Content: "Hello"}},
		}

		var got error
		for _, streamErr := range provider.CompletionStream(ctx, params) {
			got = streamErr
		}
		require.ErrorIs(t, got, context.Canceled)
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

		var got error
		for _, streamErr := range provider.CompletionStream(ctx, providers.CompletionParams{
			Model:    "test-model",
			Messages: []providers.Message{{Role: providers.RoleUser, Content: "Hello"}},
		}) {
			got = streamErr
		}
		require.ErrorIs(t, got, context.DeadlineExceeded)
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
			Messages: []providers.Message{{Role: providers.RoleUser, Content: "Hello"}},
		}

		go func() {
			time.Sleep(50 * time.Millisecond)
			cancel()
		}()

		var (
			chunkCount int
			errorCount int
			got        error
		)
		for chunk, streamErr := range provider.CompletionStream(ctx, params) {
			if streamErr != nil {
				errorCount++
				got = streamErr
				continue
			}
			if len(chunk.Choices) > 0 {
				chunkCount++
			}
		}
		require.ErrorIs(t, got, context.Canceled)
		require.Equal(t, 1, errorCount)
		require.Zero(t, chunkCount)
	})
}

func TestCompletionStreamLifecycle(t *testing.T) {
	params := providers.CompletionParams{
		Model:    "test-model",
		Messages: []providers.Message{{Role: providers.RoleUser, Content: "Hello"}},
	}

	t.Run("starts lazily", func(t *testing.T) {
		requested := make(chan struct{})
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			close(requested)
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		}))
		t.Cleanup(srv.Close)

		provider, err := NewCompatible(CompatibleConfig{
			Name:           "test-provider",
			DefaultBaseURL: srv.URL + "/v1",
			DefaultAPIKey:  "test-key",
		})
		require.NoError(t, err)

		stream := provider.CompletionStream(t.Context(), params)
		select {
		case <-requested:
			t.Fatal("CompletionStream started before iteration")
		default:
		}

		for _, streamErr := range stream {
			require.NoError(t, streamErr)
		}
		select {
		case <-requested:
		case <-time.After(time.Second):
			t.Fatal("CompletionStream did not start during iteration")
		}
	})

	t.Run("stopping iteration closes the request", func(t *testing.T) {
		requestDone := make(chan struct{})
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte(`data: {"id":"chunk","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{"content":"hello"}}]}

`))
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

		for _, streamErr := range provider.CompletionStream(t.Context(), params) {
			require.NoError(t, streamErr)
			break
		}

		select {
		case <-requestDone:
		case <-time.After(time.Second):
			t.Fatal("request remained active after iteration stopped")
		}
	})

	t.Run("parent cancellation closes the request while consumer is blocked", func(t *testing.T) {
		requestDone := make(chan struct{})
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte(`data: {"id":"chunk","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{"content":"hello"}}]}

`))
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

		ctx, cancel := context.WithCancel(context.Background())
		consumerBlocked := make(chan struct{})
		releaseConsumer := make(chan struct{})
		iterationDone := make(chan error, 1)
		go func() {
			first := true
			var got error
			for _, streamErr := range provider.CompletionStream(ctx, params) {
				if streamErr != nil {
					got = streamErr
					continue
				}
				if first {
					first = false
					close(consumerBlocked)
					<-releaseConsumer
				}
			}
			iterationDone <- got
		}()

		select {
		case <-consumerBlocked:
		case <-time.After(time.Second):
			t.Fatal("consumer did not receive the first chunk")
		}
		cancel()
		select {
		case <-requestDone:
		case <-time.After(time.Second):
			t.Fatal("parent cancellation did not close the blocked stream request")
		}
		close(releaseConsumer)
		select {
		case got := <-iterationDone:
			require.ErrorIs(t, got, context.Canceled)
		case <-time.After(time.Second):
			t.Fatal("stream iteration did not finish after cancellation")
		}
	})
}
