<div align="center">

# llm

[![Go Reference](https://pkg.go.dev/badge/github.com/humbornjo/llm.svg)](https://pkg.go.dev/github.com/humbornjo/llm)
[![Go Report Card](https://goreportcard.com/badge/github.com/humbornjo/llm)](https://goreportcard.com/report/github.com/humbornjo/llm)
![Go 1.26+](https://img.shields.io/badge/go-1.26%2B-blue.svg)

**Communicate with any LLM provider using a single, unified interface.**
Switch between OpenAI, Anthropic, Gemini, and more without changing your code.

[Documentation](docs/quickstart.md)

</div>

> [!NOTE]
> This repository is an independent downstream fork of
> [mozilla-ai/any-llm-go](https://github.com/mozilla-ai/any-llm-go), maintained
> under the module path `github.com/humbornjo/llm`. The fork preserves the
> upstream Git history and Apache License 2.0 attribution while allowing its
> API and provider abstractions to evolve for Fuss independently of upstream.
> It is not an official Mozilla AI distribution and is not endorsed by Mozilla
> AI. See [LICENSE](LICENSE) for the applicable terms.

## Quickstart

```bash
go get github.com/humbornjo/llm
export OPENAI_API_KEY="YOUR_KEY_HERE"  # or ANTHROPIC_API_KEY, etc
```

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/humbornjo/llm"
    "github.com/humbornjo/llm/providers/openai"
)

func main() {
    ctx := context.Background()

    provider, err := openai.New()
    if err != nil {
        log.Fatal(err)
    }

    response, err := provider.Completion(ctx, llm.CompletionParams{
        Model: "gpt-4o-mini",
        Messages: []llm.Message{
            {Role: llm.ROLE_USER, Content: llm.ContentFromString("Hello!")},
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(response.Choices[0].Message.ContentString())
}
```

**That's it!** To switch providers, change the import and constructor (e.g., `anthropic.New()` instead of `openai.New()`).

## Installation

### Requirements

- Go 1.26 or newer
- API keys for whichever LLM providers you want to use

Import the main package and the providers you need:

```go
import (
    "github.com/humbornjo/llm"
    "github.com/humbornjo/llm/providers/openai"    // OpenAI
    "github.com/humbornjo/llm/providers/anthropic" // Anthropic
)
```

See our [list of supported providers](docs/providers.md) to choose which ones you need.

### Setting Up API Keys

Set environment variables for your chosen providers:

```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export GEMINI_API_KEY="your-key-here"
export DEEPSEEK_API_KEY="your-key-here"
export MOONSHOT_API_KEY="your-key-here"
```

Alternatively, pass API keys directly in your code:

```go
provider, err := openai.New(llm.WithAPIKey("your-key-here"))
```

## Usage

Create a provider instance and use it for requests:

```go
import (
    "context"
    "fmt"
    "log"

    "github.com/humbornjo/llm"
    "github.com/humbornjo/llm/providers/openai"
)

provider, err := openai.New(llm.WithAPIKey("your-api-key"))
if err != nil {
    log.Fatal(err)
}

ctx := context.Background()

response, err := provider.Completion(ctx, llm.CompletionParams{
    Model: "gpt-4o-mini",
    Messages: []llm.Message{
        {Role: llm.ROLE_USER, Content: llm.ContentFromString("Hello!")},
    },
})
if err != nil {
    log.Fatal(err)
}

fmt.Println(response.Choices[0].Message.ContentString())
```

Provider instances are reusable and recommended for production applications.

### Streaming

```go
chunks, errs := provider.CompletionStream(ctx, llm.CompletionParams{
    Model: "gpt-4o-mini",
    Messages: []llm.Message{
        {Role: llm.ROLE_USER, Content: llm.ContentFromString("Write a short poem about Go.")},
    },
})

for chunk := range chunks {
    if len(chunk.Choices) > 0 {
        fmt.Print(chunk.Choices[0].Delta.Content)
    }
}

if err := <-errs; err != nil {
    log.Fatal(err)
}
```

### Tools / Function Calling

```go
response, err := provider.Completion(ctx, llm.CompletionParams{
    Model: "gpt-4o-mini",
    Messages: []llm.Message{
        {Role: llm.ROLE_USER, Content: llm.ContentFromString("What's the weather in Paris?")},
    },
    Tools: []llm.ToolInfo{
        {
            Type: "function",
            Function: llm.Function{
                Name:        "get_weather",
                Description: "Get the current weather for a location",
                Parameters: map[string]any{
                    "type": "object",
                    "properties": map[string]any{
                        "location": map[string]any{
                            "type":        "string",
                            "description": "The city name",
                        },
                    },
                    "required": []string{"location"},
                },
            },
        },
    },
    ToolChoice: "auto",
})

// Check for tool calls.
if len(response.Choices[0].Message.ToolCalls) > 0 {
    tc := response.Choices[0].Message.ToolCalls[0]
    fmt.Printf("Function: %s, Args: %s\n", tc.Function.Name, tc.Function.Arguments)
}
```

`ToolInfo` is the serializable declaration sent to an LLM provider. `Tool` is
the executable interface used by agent loops and tool dispatchers. An
implementation supplies its `Info` and `Function`, plus synchronous and
streaming execution methods. Per-call metadata can be passed with
`WithToolMetadata`; implementations apply each `ToolOption` directly to a
zero-value `ToolConfig` before execution.

### Extended Thinking (Reasoning)

For models that support extended thinking (like Claude):

```go
response, err := provider.Completion(ctx, llm.CompletionParams{
    Model: "claude-sonnet-4-20250514",
    Messages: []llm.Message{
        {Role: llm.ROLE_USER, Content: llm.ContentFromString("Solve this step by step: What is 15% of 80?")},
    },
    ReasoningEffort: llm.REASONING_EFFORT_MEDIUM,
})

if response.Choices[0].Message.Reasoning != nil {
    fmt.Println("Thinking:", response.Choices[0].Message.Reasoning.Content)
}
fmt.Println("Answer:", response.Choices[0].Message.ContentString())
```

### Embeddings

```go
provider, _ := openai.New()
result, err := provider.Embedding(ctx, llm.EmbeddingParams{
    Model: "text-embedding-3-small",
    Input: "Hello world",
})
```

### Listing Models

```go
provider, _ := openai.New()
models, err := provider.ListModels(ctx)
for _, model := range models.Data {
    fmt.Println(model.ID)
}
```

### Error Handling

All provider errors are normalized to common error types:

```go
response, err := provider.Completion(ctx, params)
if err != nil {
    switch {
    case errors.Is(err, llm.ErrRateLimit):
        // Handle rate limiting - maybe retry with backoff.
    case errors.Is(err, llm.ErrAuthentication):
        // Handle auth errors - check API key.
    case errors.Is(err, llm.ErrContextLength):
        // Handle context too long - reduce input.
    default:
        // Handle other errors.
    }
}
```

You can also use type assertions for more details:

```go
var rateLimitErr *llm.RateLimitError
if errors.As(err, &rateLimitErr) {
    fmt.Printf("Rate limited by %s (retry after %ds)\n", rateLimitErr.Provider, rateLimitErr.RetryAfter)
}
```

## Supported Providers

|  Provider  | Completion  |  Streaming  |  Tools |  Reasoning  |  Embeddings  |
|:----------:|:-----------:|:-----------:|:------:|:-----------:|:------------:|
| Anthropic  |      ✅      |      ✅      |      ✅ |      ✅      |      ❌       |
|  DeepSeek  |      ✅      |      ✅      |      ✅ |      ✅      |      ❌       |
|   Gemini   |      ✅      |      ✅      |      ✅ |      ✅      |      ✅       |
|  Moonshot  |      ✅      |      ✅      |      ✅ |      ✅      |      ❌       |
|   OpenAI   |      ✅      |      ✅      |      ✅ |      ✅      |      ✅       |

## Why choose `llm`?

- **Simple, unified interface** - Same types and patterns across all providers, switch models with just a string change
- **Developer friendly** - Full type definitions for better IDE support and clear, actionable error messages
- **Leverages official provider SDKs** - Uses `github.com/openai/openai-go` and `github.com/anthropics/anthropic-sdk-go` for maximum compatibility
- **Stays framework-agnostic** so it can be used across different projects and use cases
- **Idiomatic Go** - Follows Go conventions with proper error handling and context support
- **Streaming support** - Channel-based streaming that's natural in Go

## Development

```bash
make lint       # Run linter with auto-fix
make test       # Lint + run all tests
make test-only  # Run tests without linting
make test-unit  # Run unit tests only (skip integration)
make build      # Verify compilation
```

## Documentation

- **[Quickstart](docs/quickstart.md)** - Get up and running in minutes
- **[Supported Providers](docs/providers.md)** - Per-provider setup, models, and features
- **[API Reference](docs/api/)** - Completion, streaming, and error handling details
- **[Examples](examples/)** - Runnable example programs

## Contributing

We welcome contributions from developers of all skill levels! Open an issue to discuss changes, and see [AGENTS.md](AGENTS.md) for the coding conventions this repository follows.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
