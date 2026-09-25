package moonshot

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/humbornjo/llm/config"
	"github.com/humbornjo/llm/errors"
	"github.com/humbornjo/llm/providers"
)

const completionFixture = `{
	"id": "chatcmpl-1",
	"object": "chat.completion",
	"created": 1728000000,
	"model": "kimi-k3-highspeed",
	"choices": [{
		"index": 0,
		"message": {
			"role": "assistant",
			"content": "Blue.",
			"reasoning_content": "frames are entirely blue"
		},
		"finish_reason": "stop"
	}],
	"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
}`

const streamFixture = `data: {"id":"chunk-1","object":"chat.completion.chunk","created":1728000000,"model":"kimi-k3-highspeed","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"thinking it over"}}]}

data: {"id":"chunk-1","object":"chat.completion.chunk","created":1728000000,"model":"kimi-k3-highspeed","choices":[{"index":0,"delta":{"content":"Blue."},"finish_reason":"stop"}]}

data: [DONE]

`

// newTestServer returns a server that responds with fixture to every
// request and, when captured is not nil, records the request body.
func newTestServer(t *testing.T, contentType, fixture string, captured *string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if captured != nil {
			// The handler runs in its own goroutine; require must not
			// be used here.
			body, _ := io.ReadAll(r.Body)
			*captured = string(body)
		}
		w.Header().Set("Content-Type", contentType)
		_, _ = w.Write([]byte(fixture))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestMoonshot_New(t *testing.T) {
	// Note: Not using t.Parallel() here because a child test uses t.Setenv.

	t.Run("returns error when API key is missing", func(t *testing.T) {
		t.Setenv(_ENV_API_KEY, "")

		provider, err := New()
		require.Error(t, err)
		require.Nil(t, provider)

		var missingKeyErr *errors.MissingAPIKeyError
		require.ErrorAs(t, err, &missingKeyErr)
	})

	t.Run("creates provider with explicit API key", func(t *testing.T) {
		t.Parallel()

		provider, err := New(config.WithAPIKey("test-key"))
		require.NoError(t, err)
		require.Equal(t, _PROVIDER_NAME, provider.Name())

		caps := provider.Capabilities()
		require.True(t, caps.Completion)
		require.True(t, caps.CompletionReasoning)
		require.True(t, caps.CompletionStreaming)
		require.True(t, caps.CompletionTools)
		require.ElementsMatch(t, []providers.ContentPartType{
			providers.CONTENT_PART_TEXT,
			providers.CONTENT_PART_IMAGE_URL,
			providers.CONTENT_PART_VIDEO_URL,
		}, caps.CompletionTypes)
		require.True(t, caps.ListModels)
		require.False(t, caps.Embedding)
	})
}

func TestMoonshot_CompletionExtensions(t *testing.T) {
	t.Parallel()

	params := providers.CompletionParams{
		Model: "kimi-k3-highspeed",
		Messages: []providers.Message{
			{Role: providers.ROLE_USER, Content: providers.ContentFromString("describe this video")},
		},
	}

	t.Run("maps reasoning_content into Reasoning", func(t *testing.T) {
		t.Parallel()

		srv := newTestServer(t, "application/json", completionFixture, nil)
		provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(srv.URL+"/v1"))
		require.NoError(t, err)

		resp, err := provider.Completion(context.Background(), params)
		require.NoError(t, err)
		require.Equal(t, "Blue.", resp.Choices[0].Message.ContentString())
		require.NotNil(t, resp.Choices[0].Message.Reasoning)
		require.Equal(t, "frames are entirely blue", resp.Choices[0].Message.Reasoning.Content)
	})

	t.Run("leaves Reasoning nil when the extension is absent", func(t *testing.T) {
		t.Parallel()

		srv := newTestServer(t, "application/json", `{
			"id": "chatcmpl-1",
			"object": "chat.completion",
			"created": 1728000000,
			"model": "kimi-k3-highspeed",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "Blue."},
				"finish_reason": "stop"
			}]
		}`, nil)
		provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(srv.URL+"/v1"))
		require.NoError(t, err)

		resp, err := provider.Completion(context.Background(), params)
		require.NoError(t, err)
		require.Nil(t, resp.Choices[0].Message.Reasoning)
	})

	t.Run("sends video_url parts on the wire", func(t *testing.T) {
		t.Parallel()

		var captured string
		srv := newTestServer(t, "application/json", completionFixture, &captured)
		provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(srv.URL+"/v1"))
		require.NoError(t, err)

		_, err = provider.Completion(context.Background(), providers.CompletionParams{
			Model: "kimi-k3-highspeed",
			Messages: []providers.Message{
				{Role: providers.ROLE_USER, Content: providers.ContentFromParts(
					&providers.ContentPartText{Text: "describe this video"},
					&providers.ContentPartVideo{VideoURL: &providers.VideoURL{URL: "data:video/mp4;base64,AAAA"}},
				)},
			},
		})
		require.NoError(t, err)
		require.Contains(t, captured,
			`{"type":"video_url","video_url":{"url":"data:video/mp4;base64,AAAA"}}`)
	})
}

func TestMoonshot_CompletionStreamExtensions(t *testing.T) {
	t.Parallel()

	srv := newTestServer(t, "text/event-stream", streamFixture, nil)
	provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(srv.URL+"/v1"))
	require.NoError(t, err)

	chunks, errs := provider.CompletionStream(context.Background(), providers.CompletionParams{
		Model: "kimi-k3-highspeed",
		Messages: []providers.Message{
			{Role: providers.ROLE_USER, Content: providers.ContentFromString("describe this video")},
		},
	})

	var collected []providers.ChatCompletionChunk
	for chunk := range chunks {
		collected = append(collected, chunk)
	}
	require.NoError(t, <-errs)

	require.Len(t, collected, 2)
	require.NotNil(t, collected[0].Choices[0].Delta.Reasoning)
	require.Equal(t, "thinking it over", collected[0].Choices[0].Delta.Reasoning.Content)
	require.Nil(t, collected[1].Choices[0].Delta.Reasoning)
	require.Equal(t, "Blue.", collected[1].Choices[0].Delta.Content)
}
