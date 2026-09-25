package openai

import (
	"github.com/humbornjo/llm/config"
	"github.com/humbornjo/llm/providers"
)

// Provider configuration constants.
const (
	_DEFAULT_BASE_URL = "https://api.openai.com/v1"
	_ENV_API_KEY      = "OPENAI_API_KEY"
	_PROVIDER_NAME    = "openai"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.EmbeddingProvider  = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider implements the providers.Provider interface for OpenAI.
// It embeds CompatibleProvider which handles the OpenAI SDK integration.
type Provider struct {
	*CompatibleProvider
}

// New creates a new OpenAI provider.
func New(opts ...config.Option) (*Provider, error) {
	base, err := NewCompatible(CompatibleConfig{
		APIKeyEnvVar:   _ENV_API_KEY,
		DefaultBaseURL: _DEFAULT_BASE_URL,
		Name:           _PROVIDER_NAME,
		RequireAPIKey:  true,
		Capabilities: providers.Capabilities{
			Completion:          true,
			CompletionImage:     true,
			CompletionPDF:       false,
			CompletionReasoning: true,
			CompletionStreaming: true,
			CompletionTools:     true,
			Embedding:           true,
			ListModels:          true,
		},
	}, opts...)
	if err != nil {
		return nil, err
	}

	return &Provider{CompatibleProvider: base}, nil
}
