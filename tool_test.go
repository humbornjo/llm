package llm

import (
	"context"
	"errors"
	"iter"
	"testing"
)

type executableTool struct{}

var _ Tool = executableTool{}

func (executableTool) Info() ToolInfo {
	return ToolInfo{Type: "function", Function: Function{Name: "echo"}}
}

func (executableTool) Function() Function {
	return Function{Name: "echo"}
}

func (executableTool) Execute(context.Context, string, ...ToolOption) (string, error) {
	return "ok", nil
}

func (executableTool) ExecuteStream(context.Context, string, ...ToolOption) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		yield("ok", nil)
	}
}

func TestLLM_ToolConfig(t *testing.T) {
	metadata := map[string]any{"request_id": "request-1"}
	var config ToolConfig
	for _, opt := range []ToolOption{WithToolMetadata(metadata)} {
		opt(&config)
	}
	if config.Metadata["request_id"] != "request-1" {
		t.Fatalf("metadata = %#v", config.Metadata)
	}
}

func TestLLM_NewToolsHandler(t *testing.T) {
	type input struct {
		Value string `json:"value"`
	}
	tool := NewTool(
		ToolInfo{Type: "function", Function: Function{Name: "echo"}},
		func(_ context.Context, args input, opts ...ToolOption) (string, error) {
			var config ToolConfig
			for _, opt := range opts {
				opt(&config)
			}
			return args.Value + ":" + config.Metadata["suffix"].(string), nil
		},
		nil,
	)
	handle := NewToolsHandler(tool)

	got, err := handle(
		t.Context(), FunctionCall{Name: "echo", Arguments: `{"value":"hello"}`},
		WithToolMetadata(map[string]any{"suffix": "world"}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if got != "hello:world" {
		t.Fatalf("handle() = %q", got)
	}

	_, err = handle(t.Context(), FunctionCall{Name: "missing"})
	if !errors.Is(err, ErrToolNotFound) {
		t.Fatalf("handle(missing) error = %v", err)
	}
}
