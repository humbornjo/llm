package anyllm

import (
	"context"
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

func TestToolConfig(t *testing.T) {
	metadata := map[string]any{"request_id": "request-1"}
	var config ToolConfig
	for _, opt := range []ToolOption{WithToolMetadata(metadata)} {
		opt(&config)
	}
	if config.Metadata["request_id"] != "request-1" {
		t.Fatalf("metadata = %#v", config.Metadata)
	}
}
