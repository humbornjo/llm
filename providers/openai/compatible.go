// Package openai provides an OpenAI provider implementation for llm.
// It also exports a base provider for other OpenAI-compatible services.
package openai

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"

	"github.com/humbornjo/llm/config"
	"github.com/humbornjo/llm/errors"
	"github.com/humbornjo/llm/providers"
)

// OpenAI API error codes.
const (
	_API_CODE_CONTENT_FILTER          = "content_filter"
	_API_CODE_CONTENT_POLICY_VIOLATED = "content_policy_violation"
	_API_CODE_CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
	_API_CODE_INVALID_API_KEY         = "invalid_api_key"
	_API_CODE_MODEL_NOT_FOUND         = "model_not_found"
	_API_CODE_RATE_LIMIT_EXCEEDED     = "rate_limit_exceeded"
)

// Object type constants.
const (
	_OBJECT_CHAT_COMPLETION       = "chat.completion"
	_OBJECT_CHAT_COMPLETION_CHUNK = "chat.completion.chunk"
	_OBJECT_EMBEDDING             = "embedding"
	_OBJECT_LIST                  = "list"
	_OBJECT_MODEL                 = "model"
)

// Response format types.
const (
	_RESPONSE_FORMAT_JSON_OBJECT = "json_object"
	_RESPONSE_FORMAT_JSON_SCHEMA = "json_schema"
)

// CompatibleConfig contains the configuration for an OpenAI-compatible provider.
// Fields are ordered alphabetically.
type CompatibleConfig struct {
	// APIKeyEnvVar is the environment variable for the API key.
	APIKeyEnvVar string

	// BaseURLEnvVar is the environment variable for the base URL.
	BaseURLEnvVar string

	// Capabilities describes what the provider supports.
	Capabilities providers.Capabilities

	// ChatCompletionChunkTransform is the streaming counterpart of
	// ChatCompletionResponseTransform, applied to each chunk after
	// conversion. Nil means no transformation.
	ChatCompletionChunkTransform func(*openai.ChatCompletionChunk, *providers.ChatCompletionChunk)

	// ChatCompletionRequestTransform is an optional function that modifies the chat
	// completion request after convertParams() builds it and before it is serialized
	// to the wire. Providers that are not fully OpenAI-compatible use this to adjust
	// wire-level fields (e.g. swapping max_completion_tokens back to max_tokens).
	// The pointer refers to a locally-constructed value owned by the caller; the
	// function must not retain it beyond the call. Nil means no transformation.
	ChatCompletionRequestTransform func(*openai.ChatCompletionNewParams)

	// ChatCompletionResponseTransform is an optional function that maps provider
	// response fields beyond the OpenAI spec (e.g. Kimi's reasoning_content)
	// into the normalized completion after convertResponse() builds it. The
	// pointers refer to locally-constructed values owned by the caller; the
	// function must not retain them beyond the call. Nil means no transformation.
	ChatCompletionResponseTransform func(*openai.ChatCompletion, *providers.ChatCompletion)

	// DefaultAPIKey is used when RequireAPIKey is false (e.g., for local servers).
	DefaultAPIKey string

	// DefaultBaseURL is the default API base URL.
	DefaultBaseURL string

	// Name is the provider name used in error messages.
	Name string

	// RequireAPIKey indicates whether an API key is required.
	RequireAPIKey bool

	// RequireBaseURL indicates whether a base URL must be resolvable from
	// WithBaseURL, BaseURLEnvVar, or DefaultBaseURL. When true, NewCompatible
	// returns an error if none of those yield a value. Used by providers that
	// have no sensible default endpoint (e.g. gateway).
	RequireBaseURL bool
}

// Ensure CompatibleProvider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*CompatibleProvider)(nil)
	_ providers.EmbeddingProvider  = (*CompatibleProvider)(nil)
	_ providers.ErrorConverter     = (*CompatibleProvider)(nil)
	_ providers.ModelLister        = (*CompatibleProvider)(nil)
	_ providers.Provider           = (*CompatibleProvider)(nil)
)

// CompatibleProvider implements the providers.Provider interface for OpenAI-compatible APIs.
// It can be embedded by other providers that use OpenAI-compatible endpoints.
type CompatibleProvider struct {
	compatibleConfig CompatibleConfig
	client           openai.Client
}

// NewCompatible creates a new OpenAI-compatible provider.
func NewCompatible(compatCfg CompatibleConfig, opts ...config.Option) (*CompatibleProvider, error) {
	cfg, err := config.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}

	if validErr := validateCompatibleConfig(compatCfg); validErr != nil {
		return nil, validErr
	}

	baseURL, err := cfg.ResolveBaseURL(compatCfg.BaseURLEnvVar, compatCfg.DefaultBaseURL)
	if err != nil {
		return nil, err
	}

	if baseURL == "" && compatCfg.RequireBaseURL {
		if compatCfg.BaseURLEnvVar == "" {
			return nil, fmt.Errorf(
				"%s base URL is required (set via WithBaseURL option)",
				compatCfg.Name,
			)
		}

		return nil, fmt.Errorf(
			"%s base URL is required (set via WithBaseURL option or %q env var)",
			compatCfg.Name,
			compatCfg.BaseURLEnvVar,
		)
	}

	apiKey := resolveAPIKey(cfg, compatCfg)

	if apiKey == "" && compatCfg.RequireAPIKey {
		return nil, errors.NewMissingAPIKeyError(compatCfg.Name, compatCfg.APIKeyEnvVar)
	}
	if apiKey == "" {
		apiKey = compatCfg.DefaultAPIKey
	}

	clientOpts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithHTTPClient(cfg.HTTPClient()),
	}

	if baseURL != "" {
		clientOpts = append(clientOpts, option.WithBaseURL(baseURL))
	}

	return &CompatibleProvider{
		compatibleConfig: compatCfg,
		client:           openai.NewClient(clientOpts...),
	}, nil
}

// Capabilities returns the provider's capabilities.
func (p *CompatibleProvider) Capabilities() providers.Capabilities {
	return p.compatibleConfig.Capabilities
}

// Completion performs a chat completion request.
func (p *CompatibleProvider) Completion(
	ctx context.Context,
	params providers.CompletionParams,
) (*providers.ChatCompletion, error) {
	if err := validateCompletionParams(params); err != nil {
		return nil, err
	}

	req := convertParams(params)
	if p.compatibleConfig.ChatCompletionRequestTransform != nil {
		p.compatibleConfig.ChatCompletionRequestTransform(&req)
	}

	resp, err := p.client.Chat.Completions.New(ctx, req)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	result := convertResponse(resp)
	if p.compatibleConfig.ChatCompletionResponseTransform != nil {
		p.compatibleConfig.ChatCompletionResponseTransform(resp, result)
	}
	return result, nil
}

// CompletionStream performs a streaming chat completion request.
func (p *CompatibleProvider) CompletionStream(ctx context.Context, params providers.CompletionParams,
) (<-chan providers.ChatCompletionChunk, <-chan error) {
	chunks := make(chan providers.ChatCompletionChunk)
	errs := make(chan error, 1)

	go func() {
		defer close(chunks)
		defer close(errs)

		if err := validateCompletionParams(params); err != nil {
			errs <- err
			return
		}

		req := convertParams(params)
		if p.compatibleConfig.ChatCompletionRequestTransform != nil {
			p.compatibleConfig.ChatCompletionRequestTransform(&req)
		}
		if err := ctx.Err(); err != nil {
			errs <- err
			return
		}

		stream := p.client.Chat.Completions.NewStreaming(ctx, req)
		defer func() { _ = stream.Close() }()

		for stream.Next() {
			if err := ctx.Err(); err != nil {
				errs <- err
				return
			}
			chunk := stream.Current()
			converted := convertChunk(&chunk)
			if p.compatibleConfig.ChatCompletionChunkTransform != nil {
				p.compatibleConfig.ChatCompletionChunkTransform(&chunk, &converted)
			}
			select {
			case chunks <- converted:
			case <-ctx.Done():
				// Caller cancelled mid-stream; surface ctx.Err() so the
				// consumer can tell a cancelled stream apart from one
				// that completed cleanly, rather than seeing a bare
				// close on the error channel.
				errs <- ctx.Err()
				return
			}
		}

		if err := ctx.Err(); err != nil {
			errs <- err
		} else if err := stream.Err(); err != nil {
			errs <- p.ConvertError(err)
		}
	}()

	return chunks, errs
}

// ConvertError converts OpenAI-compatible errors to unified error types.
// Implements providers.ErrorConverter.
func (p *CompatibleProvider) ConvertError(err error) error {
	if err == nil {
		return nil
	}

	name := p.compatibleConfig.Name

	// Check for OpenAI API error type.
	var apiErr *openai.Error
	if stderrors.As(err, &apiErr) {
		return convertAPIError(name, apiErr, err)
	}

	// Network-level errors are wrapped as provider errors.
	// Note: We check for "connection refused" string as a fallback since
	// Go's net package doesn't expose typed errors for all network conditions.
	return errors.NewProviderError(name, err)
}

// Embedding generates embeddings for the given input.
func (p *CompatibleProvider) Embedding(
	ctx context.Context,
	params providers.EmbeddingParams,
) (*providers.EmbeddingResponse, error) {
	req := convertEmbeddingParams(params)

	resp, err := p.client.Embeddings.New(ctx, req)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return convertEmbeddingResponse(resp), nil
}

// ListModels returns a list of available models.
func (p *CompatibleProvider) ListModels(ctx context.Context) (*providers.ModelsResponse, error) {
	resp, err := p.client.Models.List(ctx)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	models := make([]providers.Model, 0, len(resp.Data))
	for _, model := range resp.Data {
		models = append(models, providers.Model{
			ID:      model.ID,
			Object:  _OBJECT_MODEL,
			Created: model.Created,
			OwnedBy: string(model.OwnedBy),
		})
	}

	return &providers.ModelsResponse{
		Object: _OBJECT_LIST,
		Data:   models,
	}, nil
}

// Name returns the provider name.
func (p *CompatibleProvider) Name() string {
	return p.compatibleConfig.Name
}

// convertAPIError converts an OpenAI API error to a unified error type.
func convertAPIError(name string, apiErr *openai.Error, originalErr error) error {
	switch apiErr.StatusCode {
	case 400:
		if apiErr.Code == _API_CODE_CONTEXT_LENGTH_EXCEEDED {
			return errors.NewContextLengthError(name, originalErr)
		}
		if apiErr.Code == _API_CODE_CONTENT_FILTER || apiErr.Code == _API_CODE_CONTENT_POLICY_VIOLATED {
			return errors.NewContentFilterError(name, originalErr)
		}
		return errors.NewInvalidRequestError(name, originalErr)
	case 401:
		return errors.NewAuthenticationError(name, originalErr)
	case 404:
		return errors.NewModelNotFoundError(name, originalErr)
	case 429:
		return errors.NewRateLimitError(name, originalErr)
	}

	// Check error code for additional classification.
	switch apiErr.Code {
	case _API_CODE_INVALID_API_KEY:
		return errors.NewAuthenticationError(name, originalErr)
	case _API_CODE_MODEL_NOT_FOUND:
		return errors.NewModelNotFoundError(name, originalErr)
	case _API_CODE_RATE_LIMIT_EXCEEDED:
		return errors.NewRateLimitError(name, originalErr)
	}

	return errors.NewProviderError(name, originalErr)
}

// convertAssistantMessage converts an assistant message to OpenAI format.
func convertAssistantMessage(msg providers.Message) (openai.ChatCompletionMessageParamUnion, error) {
	text, err := contentText(msg)
	if err != nil {
		return openai.ChatCompletionMessageParamUnion{}, err
	}

	if len(msg.ToolCalls) > 0 {
		toolCalls := make([]openai.ChatCompletionMessageToolCallParam, 0, len(msg.ToolCalls))
		for _, tc := range msg.ToolCalls {
			toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallParam{
				ID: tc.ID,
				Function: openai.ChatCompletionMessageToolCallFunctionParam{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			})
		}
		assistant := &openai.ChatCompletionAssistantMessageParam{
			ToolCalls: toolCalls,
		}
		// Beside tool calls the content field may be omitted, and strict
		// providers reject an explicit empty string.
		if text != "" {
			assistant.Content.OfString = openai.String(text)
		}
		return openai.ChatCompletionMessageParamUnion{
			OfAssistant: assistant,
		}, nil
	}
	return openai.AssistantMessage(text), nil
}

// contentText reduces message content to plain text for the chat wire:
// scalar content is returned as-is, and parts-backed content
// concatenates its text parts. System, tool, and assistant messages
// carry only text on the wire, so any other part kind is an error
// rather than silently dropped. ContentString alone would read
// parts-backed content as empty.
func contentText(msg providers.Message) (string, error) {
	if !msg.IsMultiModal() {
		return msg.ContentString(), nil
	}
	var text strings.Builder
	for _, part := range msg.ContentParts() {
		value := part.Unwrap()
		if value == nil {
			return "", fmt.Errorf("%s message content part must not be nil", msg.Role)
		}
		textPart, ok := value.(*providers.ContentPartText)
		if !ok {
			return "", fmt.Errorf(
				"%s message content supports only text parts, got %q",
				msg.Role, value.GetType(),
			)
		}
		text.WriteString(textPart.Text)
	}
	return text.String(), nil
}

// convertChunk converts an OpenAI streaming chunk to provider format.
func convertChunk(chunk *openai.ChatCompletionChunk) providers.ChatCompletionChunk {
	choices := make([]providers.ChunkChoice, 0, len(chunk.Choices))
	for _, choice := range chunk.Choices {
		chunkChoice := providers.ChunkChoice{
			Index: int(choice.Index),
			Delta: providers.ChunkDelta{
				Role:    string(choice.Delta.Role),
				Content: choice.Delta.Content,
			},
			FinishReason: string(choice.FinishReason),
		}

		if len(choice.Delta.ToolCalls) > 0 {
			chunkChoice.Delta.ToolCalls = make([]providers.ToolCall, 0, len(choice.Delta.ToolCalls))
			for _, tc := range choice.Delta.ToolCalls {
				chunkChoice.Delta.ToolCalls = append(chunkChoice.Delta.ToolCalls, providers.ToolCall{
					ID:   tc.ID,
					Type: string(tc.Type),
					Function: providers.FunctionCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				})
			}
		}

		choices = append(choices, chunkChoice)
	}

	result := providers.ChatCompletionChunk{
		ID:                chunk.ID,
		Object:            _OBJECT_CHAT_COMPLETION_CHUNK,
		Created:           chunk.Created,
		Model:             chunk.Model,
		Choices:           choices,
		SystemFingerprint: chunk.SystemFingerprint,
	}

	if chunk.JSON.Usage.Valid() {
		result.Usage = convertUsage(chunk.Usage)
	}

	return result
}

// convertEmbeddingParams converts provider embedding params to OpenAI format.
func convertEmbeddingParams(params providers.EmbeddingParams) openai.EmbeddingNewParams {
	req := openai.EmbeddingNewParams{
		Model: openai.EmbeddingModel(params.Model),
	}

	switch v := params.Input.(type) {
	case string:
		req.Input = openai.EmbeddingNewParamsInputUnion{
			OfString: openai.String(v),
		}
	case []string:
		req.Input = openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: v,
		}
	default:
		// For unsupported types, convert to string representation.
		req.Input = openai.EmbeddingNewParamsInputUnion{
			OfString: openai.String(fmt.Sprintf("%v", params.Input)),
		}
	}

	if params.EncodingFormat != "" {
		req.EncodingFormat = openai.EmbeddingNewParamsEncodingFormat(params.EncodingFormat)
	}

	if params.Dimensions != nil {
		req.Dimensions = openai.Int(int64(*params.Dimensions))
	}

	if params.User != "" {
		req.User = openai.String(params.User)
	}

	return req
}

// convertEmbeddingResponse converts an OpenAI embedding response to provider format.
func convertEmbeddingResponse(resp *openai.CreateEmbeddingResponse) *providers.EmbeddingResponse {
	data := make([]providers.EmbeddingData, 0, len(resp.Data))
	for _, d := range resp.Data {
		embedding := make([]float64, len(d.Embedding))
		copy(embedding, d.Embedding)
		data = append(data, providers.EmbeddingData{
			Object:    _OBJECT_EMBEDDING,
			Embedding: embedding,
			Index:     int(d.Index),
		})
	}

	result := &providers.EmbeddingResponse{
		Object: _OBJECT_LIST,
		Data:   data,
		Model:  resp.Model,
	}

	if resp.Usage.PromptTokens > 0 || resp.Usage.TotalTokens > 0 {
		result.Usage = &providers.EmbeddingUsage{
			PromptTokens: int(resp.Usage.PromptTokens),
			TotalTokens:  int(resp.Usage.TotalTokens),
		}
	}

	return result
}

// convertMessage converts a single message to OpenAI format.
func convertMessage(msg providers.Message) (openai.ChatCompletionMessageParamUnion, error) {
	switch msg.Role {
	case providers.ROLE_ASSISTANT:
		return convertAssistantMessage(msg)
	case providers.ROLE_SYSTEM:
		text, err := contentText(msg)
		if err != nil {
			return openai.ChatCompletionMessageParamUnion{}, err
		}
		return openai.SystemMessage(text), nil
	case providers.ROLE_TOOL:
		text, err := contentText(msg)
		if err != nil {
			return openai.ChatCompletionMessageParamUnion{}, err
		}
		return openai.ToolMessage(text, msg.ToolCallID), nil
	case providers.ROLE_USER:
		return convertUserMessage(msg)
	default:
		return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("unknown message role: %q", msg.Role)
	}
}

// convertMessages converts provider messages to OpenAI format.
func convertMessages(messages []providers.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	result := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages))
	for _, msg := range messages {
		converted, err := convertMessage(msg)
		if err != nil {
			return nil, err
		}
		result = append(result, converted)
	}
	return result, nil
}

// convertParams converts providers.CompletionParams to OpenAI request parameters.
func convertParams(params providers.CompletionParams) openai.ChatCompletionNewParams {
	messages, _ := convertMessages(params.Messages) // Error already checked in validateCompletionParams

	req := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel(params.Model),
		Messages: messages,
	}

	if params.Temperature != nil {
		req.Temperature = openai.Float(*params.Temperature)
	}

	if params.TopP != nil {
		req.TopP = openai.Float(*params.TopP)
	}

	if params.MaxTokens != nil {
		req.MaxCompletionTokens = openai.Int(int64(*params.MaxTokens))
	}

	if len(params.Stop) > 0 {
		req.Stop = openai.ChatCompletionNewParamsStopUnion{
			OfStringArray: params.Stop,
		}
	}

	if len(params.Tools) > 0 {
		req.Tools = convertTools(params.Tools)
	}

	if params.ToolChoice != nil {
		req.ToolChoice = convertToolChoice(params.ToolChoice)
	}

	if params.ParallelToolCalls != nil {
		req.ParallelToolCalls = openai.Bool(*params.ParallelToolCalls)
	}

	if params.ResponseFormat != nil {
		req.ResponseFormat = convertResponseFormat(params.ResponseFormat)
	}

	if params.Seed != nil {
		req.Seed = openai.Int(int64(*params.Seed))
	}

	if params.User != "" {
		req.User = openai.String(params.User)
	}

	if params.ReasoningEffort != "" && params.ReasoningEffort != providers.REASONING_EFFORT_NONE {
		req.ReasoningEffort = shared.ReasoningEffort(params.ReasoningEffort)
	}

	if params.StreamOptions != nil && params.StreamOptions.IncludeUsage {
		req.StreamOptions = openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: openai.Bool(true),
		}
	}

	return req
}

// convertResponse converts an OpenAI response to provider format.
func convertResponse(resp *openai.ChatCompletion) *providers.ChatCompletion {
	choices := make([]providers.Choice, 0, len(resp.Choices))
	for _, choice := range resp.Choices {
		choices = append(choices, providers.Choice{
			Index:        int(choice.Index),
			Message:      convertResponseMessage(choice.Message),
			FinishReason: string(choice.FinishReason),
		})
	}

	result := &providers.ChatCompletion{
		ID:                resp.ID,
		Object:            _OBJECT_CHAT_COMPLETION,
		Created:           resp.Created,
		Model:             resp.Model,
		Choices:           choices,
		SystemFingerprint: resp.SystemFingerprint,
	}

	if resp.JSON.Usage.Valid() {
		result.Usage = convertUsage(resp.Usage)
	}

	return result
}

func convertUsage(usage openai.CompletionUsage) *providers.Usage {
	result := &providers.Usage{
		TotalTokens:      int(usage.TotalTokens),
		PromptTokens:     int(usage.PromptTokens),
		CompletionTokens: int(usage.CompletionTokens),
	}
	if usage.JSON.PromptTokensDetails.Valid() {
		details := &providers.PromptTokensDetails{}
		if usage.PromptTokensDetails.JSON.AudioTokens.Valid() {
			value := int(usage.PromptTokensDetails.AudioTokens)
			details.AudioTokens = &value
		}
		if usage.PromptTokensDetails.JSON.CachedTokens.Valid() {
			value := int(usage.PromptTokensDetails.CachedTokens)
			details.CachedTokens = &value
		}
		result.PromptTokensDetails = details
	}
	if usage.JSON.CompletionTokensDetails.Valid() {
		details := &providers.CompletionTokensDetails{}
		if usage.CompletionTokensDetails.JSON.AcceptedPredictionTokens.Valid() {
			value := int(usage.CompletionTokensDetails.AcceptedPredictionTokens)
			details.AcceptedPredictionTokens = &value
		}
		if usage.CompletionTokensDetails.JSON.AudioTokens.Valid() {
			value := int(usage.CompletionTokensDetails.AudioTokens)
			details.AudioTokens = &value
		}
		if usage.CompletionTokensDetails.JSON.ReasoningTokens.Valid() {
			value := int(usage.CompletionTokensDetails.ReasoningTokens)
			details.ReasoningTokens = &value
		}
		if usage.CompletionTokensDetails.JSON.RejectedPredictionTokens.Valid() {
			value := int(usage.CompletionTokensDetails.RejectedPredictionTokens)
			details.RejectedPredictionTokens = &value
		}
		result.CompletionTokenDetails = details
	}
	return result
}

// convertResponseFormat converts provider response format to OpenAI format.
func convertResponseFormat(format *providers.ResponseFormat) openai.ChatCompletionNewParamsResponseFormatUnion {
	if format == nil {
		return openai.ChatCompletionNewParamsResponseFormatUnion{}
	}

	switch format.Type {
	case _RESPONSE_FORMAT_JSON_OBJECT:
		return openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &openai.ResponseFormatJSONObjectParam{},
		}
	case _RESPONSE_FORMAT_JSON_SCHEMA:
		if format.JSONSchema != nil {
			strict := format.JSONSchema.Strict != nil && *format.JSONSchema.Strict
			return openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{
					JSONSchema: openai.ResponseFormatJSONSchemaJSONSchemaParam{
						Name:        format.JSONSchema.Name,
						Description: openai.String(format.JSONSchema.Description),
						Schema:      format.JSONSchema.Schema,
						Strict:      openai.Bool(strict),
					},
				},
			}
		}
	}

	return openai.ChatCompletionNewParamsResponseFormatUnion{
		OfText: &openai.ResponseFormatTextParam{},
	}
}

// convertResponseMessage converts an OpenAI response message to provider format.
func convertResponseMessage(msg openai.ChatCompletionMessage) providers.Message {
	result := providers.Message{
		Role:    string(msg.Role),
		Content: providers.ContentFromString(msg.Content),
	}

	if len(msg.ToolCalls) > 0 {
		result.ToolCalls = make([]providers.ToolCall, 0, len(msg.ToolCalls))
		for _, tc := range msg.ToolCalls {
			result.ToolCalls = append(result.ToolCalls, providers.ToolCall{
				ID:   tc.ID,
				Type: string(tc.Type),
				Function: providers.FunctionCall{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			})
		}
	}

	return result
}

// convertToolChoice converts provider tool choice to OpenAI format.
func convertToolChoice(choice any) openai.ChatCompletionToolChoiceOptionUnionParam {
	switch v := choice.(type) {
	case string:
		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String(v),
		}
	case providers.ToolChoice:
		if v.Function != nil {
			return openai.ChatCompletionToolChoiceOptionParamOfChatCompletionNamedToolChoice(
				openai.ChatCompletionNamedToolChoiceFunctionParam{
					Name: v.Function.Name,
				},
			)
		}
	}
	return openai.ChatCompletionToolChoiceOptionUnionParam{
		OfAuto: openai.String("auto"),
	}
}

// convertTools converts provider tools to OpenAI format.
func convertTools(tools []providers.ToolInfo) []openai.ChatCompletionToolParam {
	result := make([]openai.ChatCompletionToolParam, 0, len(tools))
	for _, tool := range tools {
		result = append(result, openai.ChatCompletionToolParam{
			Function: openai.FunctionDefinitionParam{
				Name:        tool.Function.Name,
				Description: openai.String(tool.Function.Description),
				Parameters:  openai.FunctionParameters(tool.Function.Parameters),
			},
		})
	}
	return result
}

// convertUserMessage converts a user message to OpenAI format.
func convertUserMessage(msg providers.Message) (openai.ChatCompletionMessageParamUnion, error) {
	if !msg.IsMultiModal() {
		return openai.UserMessage(msg.ContentString()), nil
	}

	parts := make([]openai.ChatCompletionContentPartUnionParam, 0, len(msg.ContentParts()))
	for _, part := range msg.ContentParts() {
		switch value := part.Unwrap().(type) {
		case *providers.ContentPartText:
			parts = append(parts, openai.TextContentPart(value.Text))
		case *providers.ContentPartImage:
			if value.ImageURL == nil {
				return openai.ChatCompletionMessageParamUnion{}, stderrors.New(
					"image content part requires image_url",
				)
			}
			parts = append(parts, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
				URL:    value.ImageURL.URL,
				Detail: value.ImageURL.Detail,
			}))
		case *providers.ContentPartAudio:
			if value.InputAudio == nil {
				return openai.ChatCompletionMessageParamUnion{}, stderrors.New(
					"audio content part requires input_audio",
				)
			}
			parts = append(
				parts,
				openai.InputAudioContentPart(openai.ChatCompletionContentPartInputAudioInputAudioParam{
					Data:   value.InputAudio.Data,
					Format: value.InputAudio.Format,
				}),
			)
		case *providers.ContentPartFile:
			if value.File == nil {
				return openai.ChatCompletionMessageParamUnion{}, stderrors.New(
					"file content part requires file",
				)
			}
			file := openai.ChatCompletionContentPartFileFileParam{}
			if value.File.FileId != "" {
				file.FileID = openai.String(value.File.FileId)
			}
			if value.File.FileName != "" {
				file.Filename = openai.String(value.File.FileName)
			}
			if value.File.FileData != "" {
				file.FileData = openai.String(value.File.FileData)
			}
			parts = append(parts, openai.FileContentPart(file))
		case *providers.ContentPartVideo:
			if value.VideoURL == nil {
				return openai.ChatCompletionMessageParamUnion{}, stderrors.New(
					"video content part requires video_url",
				)
			}
			// The OpenAI spec has no video part; video_url is a provider
			// extension (e.g. Kimi), so emit the raw wire form through
			// the SDK's union override.
			raw, err := json.Marshal(value)
			if err != nil {
				return openai.ChatCompletionMessageParamUnion{}, err
			}
			parts = append(parts, param.Override[openai.ChatCompletionContentPartUnionParam](
				json.RawMessage(raw),
			))
		default:
			return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf(
				"unknown content part type %T", value,
			)
		}
	}
	return openai.UserMessage(parts), nil
}

// resolveAPIKey resolves the API key from config or environment.
func resolveAPIKey(cfg *config.Config, compatCfg CompatibleConfig) string {
	if compatCfg.APIKeyEnvVar != "" {
		return cfg.ResolveAPIKey(compatCfg.APIKeyEnvVar)
	}
	return cfg.APIKey
}

// validateCompatibleConfig validates the compatible provider configuration.
func validateCompatibleConfig(cfg CompatibleConfig) error {
	if cfg.Name == "" {
		return fmt.Errorf("provider name is required")
	}
	return nil
}

// validateCompletionParams validates completion parameters.
func validateCompletionParams(params providers.CompletionParams) error {
	if params.Model == "" {
		return errors.NewInvalidRequestError("", fmt.Errorf("model is required"))
	}
	if len(params.Messages) == 0 {
		return errors.NewInvalidRequestError("", fmt.Errorf("at least one message is required"))
	}

	// Validate message roles.
	for _, msg := range params.Messages {
		if _, err := convertMessage(msg); err != nil {
			return errors.NewInvalidRequestError("", err)
		}
	}

	return nil
}
