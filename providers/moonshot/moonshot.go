// Package moonshot provides a Moonshot AI (Kimi) provider implementation
// for llm. Kimi exposes an OpenAI-compatible API with extensions beyond
// the OpenAI spec: reasoning_content on responses and video_url content
// parts on requests.
package moonshot

import (
	"encoding/json"

	oaisdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/respjson"

	"github.com/humbornjo/llm/config"
	"github.com/humbornjo/llm/providers"
	"github.com/humbornjo/llm/providers/openai"
)

// Provider configuration constants.
const (
	_DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
	_ENV_API_KEY      = "MOONSHOT_API_KEY"
	_PROVIDER_NAME    = "moonshot"
)

// Response extension fields beyond the OpenAI spec.
const (
	_FIELD_REASONING_CONTENT = "reasoning_content"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider implements the providers.Provider interface for Moonshot AI.
// It embeds openai.CompatibleProvider since Kimi exposes an OpenAI-compatible API.
type Provider struct {
	*openai.CompatibleProvider
}

// New creates a new Moonshot AI provider.
func New(opts ...config.Option) (*Provider, error) {
	base, err := openai.NewCompatible(openai.CompatibleConfig{
		APIKeyEnvVar:   _ENV_API_KEY,
		BaseURLEnvVar:  "",
		DefaultAPIKey:  "",
		DefaultBaseURL: _DEFAULT_BASE_URL,
		Name:           _PROVIDER_NAME,
		RequireAPIKey:  true,
		Capabilities: providers.Capabilities{
			Completion:          true,
			CompletionImage:     true,  // Base64 data URLs only, no public URLs.
			CompletionPDF:       false, // No file content parts.
			CompletionReasoning: true,  // Kimi K3 thinking models.
			CompletionStreaming: true,
			CompletionTools:     true,
			Embedding:           false, // Kimi hosts no embedding models.
			ListModels:          true,
		},
		ChatCompletionChunkTransform:    convertChunkExtensions,
		ChatCompletionResponseTransform: convertResponseExtensions,
	}, opts...)
	if err != nil {
		return nil, err
	}

	return &Provider{CompatibleProvider: base}, nil
}

// convertChunkExtensions maps Kimi stream delta fields beyond the OpenAI
// spec into the normalized chunk.
func convertChunkExtensions(raw *oaisdk.ChatCompletionChunk, normalized *providers.ChatCompletionChunk) {
	for i := range normalized.Choices {
		if i >= len(raw.Choices) {
			break
		}
		normalized.Choices[i].Delta.Reasoning = reasoningFrom(raw.Choices[i].Delta.JSON.ExtraFields)
	}
}

// convertResponseExtensions maps Kimi response fields beyond the OpenAI
// spec into the normalized completion.
func convertResponseExtensions(raw *oaisdk.ChatCompletion, normalized *providers.ChatCompletion) {
	for i := range normalized.Choices {
		if i >= len(raw.Choices) {
			break
		}
		normalized.Choices[i].Message.Reasoning = reasoningFrom(raw.Choices[i].Message.JSON.ExtraFields)
	}
}

// reasoningFrom extracts reasoning_content from provider-specific extra
// fields, returning nil when it is absent or empty. Extra fields carry
// an invalid status because they match no schema field, so presence —
// not Valid — is the signal.
func reasoningFrom(extraFields map[string]respjson.Field) *providers.Reasoning {
	field, ok := extraFields[_FIELD_REASONING_CONTENT]
	if !ok {
		return nil
	}
	var thinking string
	if err := json.Unmarshal([]byte(field.Raw()), &thinking); err != nil || thinking == "" {
		return nil
	}
	return &providers.Reasoning{Content: thinking}
}
