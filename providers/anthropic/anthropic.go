// Package anthropic provides an Anthropic provider implementation for llm.
package anthropic

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/humbornjo/llm/config"
	"github.com/humbornjo/llm/errors"
	"github.com/humbornjo/llm/providers"
)

// Provider configuration constants.
const (
	_DEFAULT_MAX_TOKENS = 4096
	_ENV_API_KEY        = "ANTHROPIC_API_KEY"
	_ENV_BASE_URL       = "ANTHROPIC_BASE_URL"
	_PROVIDER_NAME      = "anthropic"
)

// Anthropic content block types.
const (
	_BLOCK_TYPE_TEXT     = "text"
	_BLOCK_TYPE_THINKING = "thinking"
	_BLOCK_TYPE_TOOL_USE = "tool_use"
)

// Anthropic delta types.
const (
	_DELTA_TYPE_INPUT_JSON = "input_json_delta"
	_DELTA_TYPE_TEXT       = "text_delta"
	_DELTA_TYPE_THINKING   = "thinking_delta"
)

// Anthropic error response patterns (checked in raw JSON).
const (
	_ERROR_PATTERN_CONTEXT_LENGTH = "context_length"
	_ERROR_PATTERN_TOKEN          = "token"
	_ERROR_PATTERN_CONTENT        = "content"
	_ERROR_PATTERN_SAFETY         = "safety"
)

// Anthropic streaming event types.
const (
	_EVENT_CONTENT_BLOCK_DELTA = "content_block_delta"
	_EVENT_CONTENT_BLOCK_START = "content_block_start"
	_EVENT_MESSAGE_DELTA       = "message_delta"
	_EVENT_MESSAGE_START       = "message_start"
)

// Anthropic stop reasons.
const (
	_STOP_REASON_END_TURN      = "end_turn"
	_STOP_REASON_MAX_TOKENS    = "max_tokens"
	_STOP_REASON_STOP_SEQUENCE = "stop_sequence"
	_STOP_REASON_TOOL_USE      = "tool_use"
)

// JSON schema field names.
const (
	_SCHEMA_FIELD_PROPERTIES = "properties"
	_SCHEMA_FIELD_REQUIRED   = "required"
)

// Response format types.
const (
	_RESPONSE_FORMAT_JSON_OBJECT = "json_object"
	_RESPONSE_FORMAT_JSON_SCHEMA = "json_schema"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider implements the providers.Provider interface for Anthropic.
type Provider struct {
	client *anthropic.Client
	config *config.Config
}

// streamState tracks accumulated state during streaming.
// Note: Only accessed from a single goroutine, so no synchronization needed.
type streamState struct {
	messageID      string
	model          string
	content        strings.Builder
	reasoning      strings.Builder
	toolCalls      []providers.ToolCall
	currentToolIdx int
	inputUsage     int64
	cachedUsage    int64
	cacheWrite     int64
}

// New creates a new Anthropic provider.
func New(opts ...config.Option) (*Provider, error) {
	cfg, err := config.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}

	apiKey := cfg.ResolveAPIKey(_ENV_API_KEY)
	if apiKey == "" {
		return nil, errors.NewMissingAPIKeyError(_PROVIDER_NAME, _ENV_API_KEY)
	}

	baseURL, err := cfg.ResolveBaseURL(_ENV_BASE_URL, "")
	if err != nil {
		return nil, err
	}

	clientOpts := []option.RequestOption{
		option.WithAPIKey(apiKey),
	}

	if baseURL != "" {
		clientOpts = append(clientOpts, option.WithBaseURL(baseURL))
	}

	client := anthropic.NewClient(clientOpts...)

	return &Provider{
		client: &client,
		config: cfg,
	}, nil
}

// Capabilities returns the provider's capabilities.
func (p *Provider) Capabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionImage:     true,
		CompletionPDF:       true,
		CompletionReasoning: true,
		CompletionStreaming: true,
		CompletionTools:     true,
		Embedding:           false,
		ListModels:          false,
	}
}

// Completion performs a chat completion request.
func (p *Provider) Completion(
	ctx context.Context,
	params providers.CompletionParams,
) (*providers.ChatCompletion, error) {
	req, err := p.convertParams(params)
	if err != nil {
		return nil, err
	}

	resp, err := p.client.Messages.New(ctx, req)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return convertResponse(resp), nil
}

// convertParams converts providers.CompletionParams to Anthropic request parameters.
func (p *Provider) convertParams(params providers.CompletionParams) (anthropic.MessageNewParams, error) {
	messages, system := convertMessages(params.Messages)

	maxTokens := int64(_DEFAULT_MAX_TOKENS)
	if params.MaxTokens != nil {
		maxTokens = int64(*params.MaxTokens)
	}

	req := anthropic.MessageNewParams{
		Model:     anthropic.Model(params.Model),
		Messages:  messages,
		MaxTokens: maxTokens,
	}

	if system != "" {
		req.System = []anthropic.TextBlockParam{
			{Text: system},
		}
	}

	if params.Temperature != nil {
		req.Temperature = anthropic.Float(*params.Temperature)
	}

	if params.TopP != nil {
		req.TopP = anthropic.Float(*params.TopP)
	}

	if len(params.Stop) > 0 {
		req.StopSequences = params.Stop
	}

	if len(params.Tools) > 0 {
		tools := make([]anthropic.ToolUnionParam, 0, len(params.Tools))
		for _, tool := range params.Tools {
			converted, err := convertTool(tool)
			if err != nil {
				return anthropic.MessageNewParams{}, err
			}
			tools = append(tools, converted)
		}
		req.Tools = tools
	}

	if params.ToolChoice != nil {
		req.ToolChoice = convertToolChoice(params.ToolChoice, params.ParallelToolCalls)
	}

	applyResponseFormat(&req, params.ResponseFormat)

	applyThinking(&req, params.ReasoningEffort, maxTokens)

	return req, nil
}

// CompletionStream performs a streaming chat completion request.
func (p *Provider) CompletionStream(
	ctx context.Context,
	params providers.CompletionParams,
) (<-chan providers.ChatCompletionChunk, <-chan error) {
	chunks := make(chan providers.ChatCompletionChunk)
	errs := make(chan error, 1)

	go func() {
		defer close(chunks)
		defer close(errs)

		req, err := p.convertParams(params)
		if err != nil {
			errs <- err
			return
		}

		if err := ctx.Err(); err != nil {
			errs <- err
			return
		}

		stream := p.client.Messages.NewStreaming(ctx, req)
		defer func() { _ = stream.Close() }()
		state := newStreamState()

		for stream.Next() {
			if err := ctx.Err(); err != nil {
				errs <- err
				return
			}
			event := stream.Current()

			switch event.Type {
			case _EVENT_MESSAGE_START:
				if !sendChunk(ctx, chunks, state.handleMessageStart(event.AsMessageStart())) {
					errs <- ctx.Err()
					return
				}

			case _EVENT_CONTENT_BLOCK_START:
				state.handleContentBlockStart(event.AsContentBlockStart())

			case _EVENT_CONTENT_BLOCK_DELTA:
				if chunk := state.handleContentBlockDelta(event.AsContentBlockDelta()); chunk != nil {
					if !sendChunk(ctx, chunks, *chunk) {
						errs <- ctx.Err()
						return
					}
				}

			case _EVENT_MESSAGE_DELTA:
				if !sendChunk(ctx, chunks, state.handleMessageDelta(event.AsMessageDelta())) {
					errs <- ctx.Err()
					return
				}
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

// Name returns the provider name.
func (p *Provider) Name() string {
	return _PROVIDER_NAME
}

// newStreamState creates a new stream state with default values.
func newStreamState() *streamState {
	return &streamState{
		currentToolIdx: -1,
	}
}

// chunk creates a ChatCompletionChunk with the given delta.
func (s *streamState) chunk(delta providers.ChunkDelta) providers.ChatCompletionChunk {
	return providers.ChatCompletionChunk{
		ID:     s.messageID,
		Object: "chat.completion.chunk",
		Model:  s.model,
		Choices: []providers.ChunkChoice{{
			Index: 0,
			Delta: delta,
		}},
	}
}

// handleContentBlockDelta processes a content_block_delta event and returns a chunk if applicable.
func (s *streamState) handleContentBlockDelta(event anthropic.ContentBlockDeltaEvent) *providers.ChatCompletionChunk {
	switch event.Delta.Type {
	case _DELTA_TYPE_TEXT:
		return s.handleTextDelta(event.Delta.Text)
	case _DELTA_TYPE_THINKING:
		return s.handleThinkingDelta(event.Delta.Thinking)
	case _DELTA_TYPE_INPUT_JSON:
		return s.handleInputJSONDelta(event.Delta.PartialJSON)
	default:
		return nil
	}
}

// handleContentBlockStart processes a content_block_start event.
func (s *streamState) handleContentBlockStart(event anthropic.ContentBlockStartEvent) {
	switch event.ContentBlock.Type {
	case _BLOCK_TYPE_THINKING:
		// Reasoning block started - no action needed.
	case _BLOCK_TYPE_TOOL_USE:
		s.currentToolIdx++
		// TODO: Extract to newToolCallFromBlock() if this pattern is needed elsewhere.
		tc := providers.ToolCall{
			ID:   event.ContentBlock.ID,
			Type: "function",
			Function: providers.FunctionCall{
				Name: event.ContentBlock.Name,
			},
		}
		s.toolCalls = append(s.toolCalls, tc)
	}
}

// handleInputJSONDelta processes a tool input JSON delta and returns a chunk if applicable.
func (s *streamState) handleInputJSONDelta(partialJSON string) *providers.ChatCompletionChunk {
	if s.currentToolIdx < 0 || s.currentToolIdx >= len(s.toolCalls) {
		return nil
	}

	s.toolCalls[s.currentToolIdx].Function.Arguments += partialJSON
	chunk := s.chunk(providers.ChunkDelta{
		ToolCalls: []providers.ToolCall{s.toolCalls[s.currentToolIdx]},
	})
	return &chunk
}

// handleMessageDelta processes a message_delta event and returns the final chunk.
func (s *streamState) handleMessageDelta(event anthropic.MessageDeltaEvent) providers.ChatCompletionChunk {
	finishReason := convertStopReason(string(event.Delta.StopReason))
	chunk := s.chunk(providers.ChunkDelta{})
	chunk.Choices[0].FinishReason = finishReason
	chunk.Usage = &providers.Usage{
		PromptTokens:     int(s.inputUsage),
		TotalTokens:      int(s.inputUsage + event.Usage.OutputTokens),
		CompletionTokens: int(event.Usage.OutputTokens),
		PromptTokensDetails: &providers.PromptTokensDetails{
			CachedTokens:     new(int(s.cachedUsage)),
			CacheWriteTokens: new(int(s.cacheWrite)),
		},
	}
	return chunk
}

// handleMessageStart processes a message_start event and returns the initial chunk.
func (s *streamState) handleMessageStart(event anthropic.MessageStartEvent) providers.ChatCompletionChunk {
	s.messageID = event.Message.ID
	s.model = string(event.Message.Model)
	s.inputUsage = event.Message.Usage.InputTokens
	s.cachedUsage = event.Message.Usage.CacheReadInputTokens
	s.cacheWrite = event.Message.Usage.CacheCreationInputTokens

	return s.chunk(providers.ChunkDelta{Role: providers.ROLE_ASSISTANT})
}

// handleThinkingDelta processes a thinking delta and returns a chunk.
func (s *streamState) handleThinkingDelta(thinking string) *providers.ChatCompletionChunk {
	s.reasoning.WriteString(thinking)
	chunk := s.chunk(providers.ChunkDelta{
		Reasoning: &providers.Reasoning{Content: thinking},
	})
	return &chunk
}

// handleTextDelta processes a text delta and returns a chunk.
func (s *streamState) handleTextDelta(text string) *providers.ChatCompletionChunk {
	s.content.WriteString(text)
	chunk := s.chunk(providers.ChunkDelta{Content: text})
	return &chunk
}

// applyThinking configures thinking/reasoning on the request if applicable.
func applyThinking(req *anthropic.MessageNewParams, effort providers.ReasoningEffort, maxTokens int64) {
	if effort == "" || effort == providers.REASONING_EFFORT_NONE {
		return
	}

	budget, ok := thinkingBudget(effort)
	if !ok {
		return
	}

	req.Thinking = anthropic.ThinkingConfigParamOfEnabled(budget)

	// Increase max tokens to accommodate thinking.
	minTokens := budget * 2
	if maxTokens < minTokens {
		req.MaxTokens = minTokens
	}
}

// applyResponseFormat configures structured output on the request if applicable.
func applyResponseFormat(req *anthropic.MessageNewParams, format *providers.ResponseFormat) {
	if format == nil || format.JSONSchema == nil {
		return
	}
	switch format.Type {
	case _RESPONSE_FORMAT_JSON_SCHEMA:
		// JSONOutputFormatParam only carries Schema and Type; Name, Description, and Strict
		// from providers.JSONSchema are not supported by the Anthropic API.
		req.OutputConfig = anthropic.OutputConfigParam{
			Format: anthropic.JSONOutputFormatParam{
				Schema: format.JSONSchema.Schema,
			},
		}
	case _RESPONSE_FORMAT_JSON_OBJECT:
		// Anthropic requires a schema for structured output; json_object without a schema
		// is not supported. No-op to preserve forward compatibility.
	}
}

// convertAssistantMessage converts an assistant message to Anthropic format.
func convertAssistantMessage(msg providers.Message) *anthropic.MessageParam {
	if len(msg.ToolCalls) == 0 {
		m := anthropic.NewAssistantMessage(anthropic.NewTextBlock(msg.ContentString()))
		return &m
	}

	content := make([]anthropic.ContentBlockParamUnion, 0)
	if msg.ContentString() != "" {
		content = append(content, anthropic.NewTextBlock(msg.ContentString()))
	}

	for _, tc := range msg.ToolCalls {
		content = append(content, convertToolCall(tc))
	}

	m := anthropic.NewAssistantMessage(content...)
	return &m
}

// convertImagePart converts an image URL to Anthropic format.
func convertImagePart(img *providers.ImageURL) anthropic.ContentBlockParamUnion {
	url := img.URL

	// Check if it's a base64 data URL.
	if strings.HasPrefix(url, "data:") {
		// Parse data URL: data:image/jpeg;base64,<data>.
		parts := strings.SplitN(url, ",", 2)
		if len(parts) == 2 {
			// Extract media type from the first part.
			mediaTypePart := strings.TrimPrefix(parts[0], "data:")
			mediaType := strings.Split(mediaTypePart, ";")[0]
			data := parts[1]

			return anthropic.NewImageBlockBase64(mediaType, data)
		}
	}

	// Regular URL.
	return anthropic.NewImageBlock(anthropic.URLImageSourceParam{URL: url})
}

// convertMessage converts a single message to Anthropic format.
func convertMessage(msg providers.Message) *anthropic.MessageParam {
	switch msg.Role {
	case providers.ROLE_USER:
		return convertUserMessage(msg)
	case providers.ROLE_ASSISTANT:
		return convertAssistantMessage(msg)
	case providers.ROLE_TOOL:
		return convertToolMessage(msg)
	default:
		return nil
	}
}

// convertMessages converts providers messages to Anthropic format.
// Returns the messages and the combined system message.
func convertMessages(messages []providers.Message) ([]anthropic.MessageParam, string) {
	result := make([]anthropic.MessageParam, 0, len(messages))
	var systemParts []string

	for _, msg := range messages {
		if msg.Role == providers.ROLE_SYSTEM {
			systemParts = append(systemParts, msg.ContentString())
			continue
		}

		if converted := convertMessage(msg); converted != nil {
			result = append(result, *converted)
		}
	}

	return result, strings.Join(systemParts, "\n")
}

// convertResponse converts an Anthropic response to providers format.
func convertResponse(resp *anthropic.Message) *providers.ChatCompletion {
	var content string
	var reasoning *providers.Reasoning
	var toolCalls []providers.ToolCall

	for _, block := range resp.Content {
		switch block.Type {
		case _BLOCK_TYPE_TEXT:
			content += block.Text
		case _BLOCK_TYPE_THINKING:
			reasoning = &providers.Reasoning{
				Content: block.Thinking,
			}
		case _BLOCK_TYPE_TOOL_USE:
			inputJSON := ""
			if block.Input != nil {
				if inputBytes, err := json.Marshal(block.Input); err == nil {
					inputJSON = string(inputBytes)
				}
			}
			toolCalls = append(toolCalls, providers.ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: providers.FunctionCall{
					Name:      block.Name,
					Arguments: inputJSON,
				},
			})
		}
	}

	message := providers.Message{
		Role:      providers.ROLE_ASSISTANT,
		Content:   providers.ContentFromString(content),
		ToolCalls: toolCalls,
		Reasoning: reasoning,
	}

	finishReason := convertStopReason(string(resp.StopReason))

	return &providers.ChatCompletion{
		ID:     resp.ID,
		Object: "chat.completion",
		Model:  string(resp.Model),
		Choices: []providers.Choice{{
			Index:        0,
			Message:      message,
			FinishReason: finishReason,
		}},
		Usage: &providers.Usage{
			PromptTokens:     int(resp.Usage.InputTokens),
			CompletionTokens: int(resp.Usage.OutputTokens),
			TotalTokens:      int(resp.Usage.InputTokens + resp.Usage.OutputTokens),
			PromptTokensDetails: &providers.PromptTokensDetails{
				CachedTokens:     new(int(resp.Usage.CacheReadInputTokens)),
				CacheWriteTokens: new(int(resp.Usage.CacheCreationInputTokens)),
			},
		},
	}
}

// convertStopReason converts Anthropic stop reason to OpenAI finish reason.
func convertStopReason(reason string) string {
	switch reason {
	case _STOP_REASON_END_TURN:
		return providers.FINISH_REASON_STOP
	case _STOP_REASON_MAX_TOKENS:
		return providers.FINISH_REASON_LENGTH
	case _STOP_REASON_TOOL_USE:
		return providers.FINISH_REASON_TOOL_CALLS
	case _STOP_REASON_STOP_SEQUENCE:
		return providers.FINISH_REASON_STOP
	default:
		return providers.FINISH_REASON_STOP
	}
}

// convertTool converts a providers.ToolInfo to Anthropic format.
func convertTool(tool providers.ToolInfo) (anthropic.ToolUnionParam, error) {
	inputSchema := anthropic.ToolInputSchemaParam{
		Type: "object",
	}

	if tool.Function.Parameters == nil {
		return buildToolParam(tool, inputSchema), nil
	}

	if props, ok := tool.Function.Parameters[_SCHEMA_FIELD_PROPERTIES]; ok {
		inputSchema.Properties = props
	}

	req, ok := tool.Function.Parameters[_SCHEMA_FIELD_REQUIRED]
	if !ok {
		return buildToolParam(tool, inputSchema), nil
	}

	required, err := toStringSlice(req)
	if err != nil {
		return anthropic.ToolUnionParam{}, fmt.Errorf(
			"tool %s: invalid required field: %w",
			tool.Function.Name,
			err,
		)
	}
	inputSchema.Required = required

	return buildToolParam(tool, inputSchema), nil
}

// buildToolParam constructs the final ToolUnionParam from tool metadata and schema.
func buildToolParam(tool providers.ToolInfo, schema anthropic.ToolInputSchemaParam) anthropic.ToolUnionParam {
	return anthropic.ToolUnionParam{
		OfTool: &anthropic.ToolParam{
			Name:        tool.Function.Name,
			Description: anthropic.String(tool.Function.Description),
			InputSchema: schema,
		},
	}
}

// convertToolCall converts a tool call to Anthropic content block format.
func convertToolCall(tc providers.ToolCall) anthropic.ContentBlockParamUnion {
	var input map[string]any
	_ = json.Unmarshal([]byte(tc.Function.Arguments), &input) // Ignore error: use nil on failure.

	return anthropic.ContentBlockParamUnion{
		OfToolUse: &anthropic.ToolUseBlockParam{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: input,
		},
	}
}

// convertToolChoice converts providers tool choice to Anthropic format.
func convertToolChoice(choice any, parallelToolCalls *bool) anthropic.ToolChoiceUnionParam {
	disableParallel := parallelToolCalls != nil && !*parallelToolCalls

	switch v := choice.(type) {
	case string:
		switch v {
		case "auto":
			return anthropic.ToolChoiceUnionParam{
				OfAuto: &anthropic.ToolChoiceAutoParam{
					DisableParallelToolUse: anthropic.Bool(disableParallel),
				},
			}
		case "none":
			return anthropic.ToolChoiceUnionParam{
				OfNone: &anthropic.ToolChoiceNoneParam{},
			}
		case "required", "any":
			return anthropic.ToolChoiceUnionParam{
				OfAny: &anthropic.ToolChoiceAnyParam{
					DisableParallelToolUse: anthropic.Bool(disableParallel),
				},
			}
		}
	case providers.ToolChoice:
		if v.Function != nil {
			return anthropic.ToolChoiceUnionParam{
				OfTool: &anthropic.ToolChoiceToolParam{
					Name:                   v.Function.Name,
					DisableParallelToolUse: anthropic.Bool(disableParallel),
				},
			}
		}
	}

	return anthropic.ToolChoiceUnionParam{
		OfAuto: &anthropic.ToolChoiceAutoParam{
			DisableParallelToolUse: anthropic.Bool(disableParallel),
		},
	}
}

// convertToolMessage converts a tool result message to Anthropic format.
func convertToolMessage(msg providers.Message) *anthropic.MessageParam {
	m := anthropic.NewUserMessage(
		anthropic.NewToolResultBlock(msg.ToolCallID, msg.ContentString(), false),
	)
	return &m
}

// convertUserMessage converts a user message to Anthropic format.
func convertUserMessage(msg providers.Message) *anthropic.MessageParam {
	if !msg.IsMultiModal() {
		m := anthropic.NewUserMessage(anthropic.NewTextBlock(msg.ContentString()))
		return &m
	}

	content := make([]anthropic.ContentBlockParamUnion, 0)
	for _, part := range msg.ContentParts() {
		switch part := part.Unwrap().(type) {
		case *providers.ContentPartText:
			content = append(content, anthropic.NewTextBlock(part.Text))
		case *providers.ContentPartImage:
			if part.ImageURL != nil {
				content = append(content, convertImagePart(part.ImageURL))
			}
		}
	}
	m := anthropic.NewUserMessage(content...)
	return &m
}

// sendChunk delivers a chunk on the chunks channel. It returns false if the
// context was canceled while waiting, meaning the consumer has moved on and
// the stream should terminate.
func sendChunk(
	ctx context.Context,
	chunks chan<- providers.ChatCompletionChunk,
	chunk providers.ChatCompletionChunk,
) bool {
	select {
	case chunks <- chunk:
		return true
	case <-ctx.Done():
		return false
	}
}

// thinkingBudget returns the token budget for the given reasoning effort.
// Returns the budget and true if the effort level is supported, or 0 and false otherwise.
func thinkingBudget(effort providers.ReasoningEffort) (int64, bool) {
	switch effort {
	case providers.REASONING_EFFORT_LOW:
		return 1024, true
	case providers.REASONING_EFFORT_MEDIUM:
		return 4096, true
	case providers.REASONING_EFFORT_HIGH:
		return 16384, true
	default:
		return 0, false
	}
}

// toStringSlice converts a value to []string.
// Accepts []string (returned as-is) or []any (each element must be string).
func toStringSlice(v any) ([]string, error) {
	switch typed := v.(type) {
	case []string:
		return typed, nil
	case []any:
		result := make([]string, len(typed))
		for i, elem := range typed {
			s, ok := elem.(string)
			if !ok {
				return nil, fmt.Errorf("element %d: expected string, got %T", i, elem)
			}
			result[i] = s
		}
		return result, nil
	default:
		return nil, fmt.Errorf("expected []string or []any, got %T", v)
	}
}

// ConvertError converts an Anthropic SDK error to a unified error type.
// Implements providers.ErrorConverter.
func (p *Provider) ConvertError(err error) error {
	if err == nil {
		return nil
	}

	// Extract the Anthropic API error type from the error chain.
	// If it's not an API error (e.g., network error), wrap as generic provider error.
	var apiErr *anthropic.Error
	if !stderrors.As(err, &apiErr) {
		return errors.NewProviderError(_PROVIDER_NAME, err)
	}

	// Classify by HTTP status code.
	switch apiErr.StatusCode {
	case 401:
		return errors.NewAuthenticationError(_PROVIDER_NAME, err)
	case 429:
		return errors.NewRateLimitError(_PROVIDER_NAME, err)
	case 404:
		return errors.NewModelNotFoundError(_PROVIDER_NAME, err)
	case 400:
		// Anthropic uses 400 for various client errors.
		// Check the raw JSON for context length indicators.
		rawJSON := apiErr.RawJSON()
		if strings.Contains(rawJSON, _ERROR_PATTERN_CONTEXT_LENGTH) || strings.Contains(rawJSON, _ERROR_PATTERN_TOKEN) {
			return errors.NewContextLengthError(_PROVIDER_NAME, err)
		}
		return errors.NewInvalidRequestError(_PROVIDER_NAME, err)
	case 403:
		// Forbidden - could be content filter or permission issue.
		rawJSON := apiErr.RawJSON()
		if strings.Contains(rawJSON, _ERROR_PATTERN_CONTENT) || strings.Contains(rawJSON, _ERROR_PATTERN_SAFETY) {
			return errors.NewContentFilterError(_PROVIDER_NAME, err)
		}
		return errors.NewAuthenticationError(_PROVIDER_NAME, err)
	default:
		return errors.NewProviderError(_PROVIDER_NAME, err)
	}
}
