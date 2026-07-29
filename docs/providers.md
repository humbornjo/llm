# Supported Providers

any-llm-go supports multiple LLM providers through a unified interface. Each provider is implemented as a separate package.

## Provider Status

| Provider                | ID          | Completion | Streaming | Tools | Reasoning | Embeddings | List Models |
|-------------------------|:------------|:----------:|:---------:|:-----:|:---------:|:----------:|:-----------:|
| [Anthropic](#anthropic) | `anthropic` |     ✅      |     ✅     |   ✅   |     ✅     |     ❌      |      ❌      |
| [DeepSeek](#deepseek)   | `deepseek`  |     ✅      |     ✅     |   ✅   |     ✅     |     ❌      |      ✅      |
| [Gemini](#gemini)       | `gemini`    |     ✅      |     ✅     |   ✅   |     ✅     |     ✅      |      ✅      |
| [Groq](#groq)           | `groq`      |     ✅      |     ✅     |   ✅   |     ❌     |     ❌      |      ✅      |
| [OpenAI](#openai)       | `openai`    |     ✅      |     ✅     |   ✅   |     ✅     |     ✅      |      ✅      |
| [z.ai](#zai)            | `zai`       |     ✅      |     ✅     |   ✅   |     ✅     |     ❌      |      ✅      |

### Legend

- **Completion** - Basic chat completion support
- **Streaming** - Real-time streaming responses
- **Tools** - Function calling / tool use
- **Reasoning** - Extended thinking (e.g., Claude's thinking, OpenAI o1 reasoning)
- **Embeddings** - Text embedding generation
- **List Models** - API to list available models

## Provider Details

### Anthropic

```go
import (
    anyllm "github.com/humbornjo/llm"
    "github.com/humbornjo/llm/providers/anthropic"
)

// Using environment variable (ANTHROPIC_API_KEY).
provider, err := anthropic.New()

// Or with explicit API key.
provider, err := anthropic.New(anyllm.WithAPIKey("sk-ant-..."))
```

**Environment Variable:** `ANTHROPIC_API_KEY`

**Popular Models:**
- `claude-sonnet-4-20250514` - Latest Sonnet model
- `claude-3-5-sonnet-latest` - Previous Sonnet
- `claude-3-5-haiku-latest` - Fast and cost-effective
- `claude-3-opus-latest` - Most capable (legacy)

**Extended Thinking:**

Anthropic's Claude models support extended thinking for complex reasoning tasks:

```go
response, err := provider.Completion(ctx, anyllm.CompletionParams{
    Model: "claude-sonnet-4-20250514",
    Messages: messages,
    ReasoningEffort: anyllm.ReasoningEffortMedium, // low, medium, or high
})

// Access the thinking content.
if response.Choices[0].Message.Reasoning != nil {
    fmt.Println("Thinking:", response.Choices[0].Message.Reasoning.Content)
}
```

### DeepSeek

```go
import (
    anyllm "github.com/humbornjo/llm"
    "github.com/humbornjo/llm/providers/deepseek"
)

// Using environment variable (DEEPSEEK_API_KEY).
provider, err := deepseek.New()

// Or with explicit API key.
provider, err := deepseek.New(anyllm.WithAPIKey("sk-..."))
```

**Environment Variable:** `DEEPSEEK_API_KEY`

**Popular Models:**
- `deepseek-chat` - General-purpose chat model
- `deepseek-reasoner` - Reasoning model (DeepSeek R1)

**Reasoning/Thinking:**

DeepSeek R1 supports extended thinking for complex reasoning tasks:

```go
response, err := provider.Completion(ctx, anyllm.CompletionParams{
    Model: "deepseek-reasoner",
    Messages: messages,
    ReasoningEffort: anyllm.ReasoningEffortMedium,
})

if response.Choices[0].Message.Reasoning != nil {
    fmt.Println("Thinking:", response.Choices[0].Message.Reasoning.Content)
}
```

**JSON Schema:**

DeepSeek doesn't support `json_schema` response format directly. The provider automatically handles this by injecting the schema into the user message and using `json_object` mode instead.

### Gemini

```go
import (
    anyllm "github.com/humbornjo/llm"
    "github.com/humbornjo/llm/providers/gemini"
)

// Using environment variable (GEMINI_API_KEY or GOOGLE_API_KEY).
provider, err := gemini.New()

// Or with explicit API key.
provider, err := gemini.New(anyllm.WithAPIKey("your-key"))
```

**Environment Variables:** `GEMINI_API_KEY` or `GOOGLE_API_KEY`

**Popular Models:**
- `gemini-2.5-flash` - Fast and cost-effective
- `gemini-2.5-pro` - Most capable model
- `gemini-3-flash-preview` - Reasoning-capable model

**Embedding Models:**
- `gemini-embedding-001` - Text embeddings

**Reasoning/Thinking:**

Gemini models support extended thinking for complex reasoning tasks:

```go
response, err := provider.Completion(ctx, anyllm.CompletionParams{
    Model: "gemini-3-flash-preview",
    Messages: messages,
    ReasoningEffort: anyllm.ReasoningEffortMedium, // low, medium, or high
})

// Access the thinking content.
if response.Choices[0].Message.Reasoning != nil {
    fmt.Println("Thinking:", response.Choices[0].Message.Reasoning.Content)
}
```

### Groq

Groq provides fast inference through their cloud API. It exposes an OpenAI-compatible API.

```go
import (
    anyllm "github.com/humbornjo/llm"
    "github.com/humbornjo/llm/providers/groq"
)

// Using environment variable (GROQ_API_KEY).
provider, err := groq.New()

// Or with explicit API key.
provider, err := groq.New(anyllm.WithAPIKey("gsk_..."))
```

**Environment Variable:** `GROQ_API_KEY`

**Popular Models:**
- `llama-3.1-8b-instant` - Fast and cost-effective
- `llama-3.3-70b-versatile` - More capable model
- `mixtral-8x7b-32768` - Mixtral with 32k context

**Completion:**

```go
provider, _ := groq.New()
resp, err := provider.Completion(ctx, anyllm.CompletionParams{
    Model: "llama-3.1-8b-instant",
    Messages: []anyllm.Message{
        {Role: anyllm.RoleUser, Content: "Hello!"},
    },
})
```

### OpenAI

```go
import (
    anyllm "github.com/humbornjo/llm"
    "github.com/humbornjo/llm/providers/openai"
)

// Using environment variable (OPENAI_API_KEY).
provider, err := openai.New()

// Or with explicit API key.
provider, err := openai.New(anyllm.WithAPIKey("sk-..."))

// Or with custom base URL (for Azure, proxies, etc.).
provider, err := openai.New(
    anyllm.WithAPIKey("your-key"),
    anyllm.WithBaseURL("https://your-endpoint.openai.azure.com"),
)
```

**Environment Variable:** `OPENAI_API_KEY`

**Popular Models:**
- `gpt-4o` - Most capable model
- `gpt-4o-mini` - Fast and cost-effective
- `gpt-4-turbo` - Previous generation flagship
- `o1-preview` - Reasoning model
- `o1-mini` - Smaller reasoning model

**Embedding Models:**
- `text-embedding-3-small` - Cost-effective embeddings
- `text-embedding-3-large` - Higher quality embeddings

### z.ai

z.ai provides access to the GLM model family through an OpenAI-compatible API.

```go
import (
    anyllm "github.com/humbornjo/llm"
    "github.com/humbornjo/llm/providers/zai"
)

// Using environment variable (ZAI_API_KEY).
provider, err := zai.New()

// Or with explicit API key.
provider, err := zai.New(anyllm.WithAPIKey("your-key"))
```

**Environment Variable:** `ZAI_API_KEY`

**Popular Models:**
- `glm-4.5-air` - Fast and cost-effective
- `glm-4.5` - Capable general model
- `glm-4.6` - Vision-capable model
- `glm-4.7` - Advanced model
- `glm-5` - Most capable model

**Completion:**

```go
provider, _ := zai.New()
resp, err := provider.Completion(ctx, anyllm.CompletionParams{
    Model: "glm-4.6",
    Messages: []anyllm.Message{
        {Role: anyllm.RoleUser, Content: "Hello!"},
    },
})
```

## Coming Soon

The following providers are planned for future releases:

| Provider     | Status                                            |
|--------------|---------------------------------------------------|
| Cohere       | Planned                                           |
| Together AI  | Planned                                           |
| AWS Bedrock  | Planned                                           |
| Azure OpenAI | Planned (use OpenAI with custom base URL for now) |

## Adding a New Provider

Want to add support for a new provider? See our [Contributing Guide](../CONTRIBUTING.md) for instructions on implementing a new provider.

The basic requirements are:

1. Implement the `Provider` interface
2. Use the official provider SDK when available
3. Normalize responses to OpenAI format
4. Add comprehensive tests
5. Document the provider in this file

## Provider-Specific Notes

### Response Format

All providers normalize their responses to OpenAI's format:

```go
type ChatCompletion struct {
    ID      string   `json:"id"`
    Object  string   `json:"object"`
    Created int64    `json:"created"`
    Model   string   `json:"model"`
    Choices []Choice `json:"choices"`
    Usage   *Usage   `json:"usage,omitempty"`
}
```

This means you can write provider-agnostic code that works with any supported provider.

### Error Handling

Provider-specific errors are normalized to common error types:

| Error Type | Description |
|------------|-------------|
| `ErrRateLimit` | Rate limit exceeded |
| `ErrAuthentication` | Invalid or missing API key |
| `ErrInvalidRequest` | Malformed request |
| `ErrContextLength` | Input exceeds model's context window |
| `ErrContentFilter` | Content blocked by safety filters |
| `ErrModelNotFound` | Requested model doesn't exist |

See [Error Handling](api/errors.md) for more details.
