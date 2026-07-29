package anyllm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
)

var ErrToolNotFound = errors.New("tool not found")

// Tool describes and executes one function that an LLM may call.
type Tool interface {
	Info() ToolInfo
	Function() Function
	Execute(context.Context, string, ...ToolOption) (string, error)
	ExecuteStream(context.Context, string, ...ToolOption) iter.Seq2[string, error]
}

// ToolOption configures one tool execution.
type ToolOption func(*ToolConfig)

// ToolConfig contains metadata associated with one tool execution.
type ToolConfig struct {
	Metadata map[string]any
}

// WithToolMetadata attaches protocol-specific metadata to a tool execution.
func WithToolMetadata(metadata map[string]any) ToolOption {
	return func(config *ToolConfig) {
		config.Metadata = metadata
	}
}

type tool[T any] struct {
	ToolInfo
	execf   func(ctx context.Context, args T, opts ...ToolOption) (string, error)
	streamf func(ctx context.Context, args T, opts ...ToolOption) iter.Seq2[string, error]
}

func (t *tool[T]) Info() ToolInfo {
	return t.ToolInfo
}

func (t *tool[T]) Function() Function {
	return t.ToolInfo.Function
}

func (t *tool[T]) Execute(ctx context.Context, args string, opts ...ToolOption) (string, error) {
	var input T
	if err := json.Unmarshal([]byte(args), &input); err != nil {
		return "", err
	}

	return t.execf(ctx, input, opts...)
}

func (t *tool[T]) ExecuteStream(ctx context.Context, args string, opts ...ToolOption) iter.Seq2[string, error] {
	var input T
	if err := json.Unmarshal([]byte(args), &input); err != nil {
		return func(yield func(string, error) bool) {
			yield("", err)
		}
	}

	return t.streamf(ctx, input, opts...)
}

func NewTool[T any](
	info ToolInfo,
	execf func(ctx context.Context, args T, opts ...ToolOption) (string, error),
	streamf func(ctx context.Context, args T, opts ...ToolOption) iter.Seq2[string, error],
) Tool {
	return &tool[T]{ToolInfo: info, execf: execf, streamf: streamf}
}

func NewToolsHandler(tools ...Tool) func(context.Context, FunctionCall, ...ToolOption) (string, error) {
	type handler = func(context.Context, string, ...ToolOption) (string, error)

	dispatcher := make(map[string]handler, len(tools))
	for _, tool := range tools {
		dispatcher[tool.Function().Name] = tool.Execute
	}

	return func(ctx context.Context, call FunctionCall, opts ...ToolOption) (string, error) {
		handle, ok := dispatcher[call.Name]
		if !ok {
			return "", fmt.Errorf("%w: %s", ErrToolNotFound, call.Name)
		}
		return handle(ctx, call.Arguments, opts...)
	}
}
