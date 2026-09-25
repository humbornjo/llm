# Supported Providers

llm supports multiple LLM providers through a unified interface. Each provider is implemented as a separate package.

## Provider Status

| Provider                | ID          | Completion | Streaming | Tools | Reasoning | Embeddings | List Models |
|-------------------------|:------------|:----------:|:---------:|:-----:|:---------:|:----------:|:-----------:|
| [Anthropic](#anthropic) | `anthropic` |     ✅      |     ✅     |   ✅   |     ✅     |     ❌      |      ❌      |
| [DeepSeek](#deepseek)   | `deepseek`  |     ✅      |     ✅     |   ✅   |     ✅     |     ❌      |      ✅      |
| [Gemini](#gemini)       | `gemini`    |     ✅      |     ✅     |   ✅   |     ✅     |     ✅      |      ✅      |
| [Moonshot](#moonshot)   | `moonshot`  |     ✅      |     ✅     |   ✅   |     ✅     |     ❌      |      ✅      |
| [OpenAI](#openai)       | `openai`    |     ✅      |     ✅     |   ✅   |     ✅     |     ✅      |      ✅      |

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
    "github.com/humbornjo/llm"
    "github.com/humbornjo/llm/providers/anthropic"
)

// Using environment variable (ANTHROPIC_API_KEY).
provider, err := anthropic.New()

// Or with explicit API key.
provider, err := anthropic.New(llm.WithAPIKey("sk-ant-..."))
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
response, err := provider.Completion(ctx, llm.CompletionParams{
    Model: "claude-sonnet-4-20250514",
    Messages: messages,
    ReasoningEffort: llm.REASONING_EFFORT_MEDIUM, // low, medium, or high
})

// Access the thinking content.
if response.Choices[0].Message.Reasoning != nil {
    fmt.Println("Thinking:", response.Choices[0].Message.Reasoning.Content)
}
```

### DeepSeek

```go
import (
    "github.com/humbornjo/llm"
    "github.com/humbornjo/llm/providers/deepseek"
)

// Using environment variable (DEEPSEEK_API_KEY).
provider, err := deepseek.New()

// Or with explicit API key.
provider, err := deepseek.New(llm.WithAPIKey("sk-..."))
```

**Environment Variable:** `DEEPSEEK_API_KEY`

**Popular Models:**
- `deepseek-chat` - General-purpose chat model
- `deepseek-reasoner` - Reasoning model (DeepSeek R1)

**Reasoning/Thinking:**

DeepSeek's `deepseek-reasoner` model performs extended thinking server-side; enable it with `ReasoningEffort`:

```go
response, err := provider.Completion(ctx, llm.CompletionParams{
    Model: "deepseek-reasoner",
    Messages: messages,
    ReasoningEffort: llm.REASONING_EFFORT_MEDIUM,
})
```

**JSON Schema:**

DeepSeek doesn't support `json_schema` response format directly. The provider automatically handles this by injecting the schema into the user message and using `json_object` mode instead.

### Gemini

```go
import (
    "github.com/humbornjo/llm"
    "github.com/humbornjo/llm/providers/gemini"
)

// Using environment variable (GEMINI_API_KEY or GOOGLE_API_KEY).
provider, err := gemini.New()

// Or with explicit API key.
provider, err := gemini.New(llm.WithAPIKey("your-key"))
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
response, err := provider.Completion(ctx, llm.CompletionParams{
    Model: "gemini-3-flash-preview",
    Messages: messages,
    ReasoningEffort: llm.REASONING_EFFORT_MEDIUM, // low, medium, or high
})

// Access the thinking content.
if response.Choices[0].Message.Reasoning != nil {
    fmt.Println("Thinking:", response.Choices[0].Message.Reasoning.Content)
}
```

### Moonshot

Moonshot AI's Kimi exposes an OpenAI-compatible API with extensions beyond the OpenAI spec: thinking content is returned as `reasoning_content`, and user messages accept `video_url` content parts. The provider normalizes both — thinking lands in `Message.Reasoning`, and video parts pass through on the wire.

```go
import (
    "github.com/humbornjo/llm"
    "github.com/humbornjo/llm/providers/moonshot"
)

// Using environment variable (MOONSHOT_API_KEY).
provider, err := moonshot.New()

// Or with explicit API key.
provider, err := moonshot.New(llm.WithAPIKey("sk-..."))
```

**Environment Variable:** `MOONSHOT_API_KEY`

**Popular Models:**
- `kimi-k3-highspeed` - Fast thinking model

See the [Kimi docs](https://platform.kimi.ai/docs) for the full model list.

**Reasoning/Thinking:**

Kimi K3 models return their thinking as `reasoning_content`, normalized into `Reasoning`:

```go
response, err := provider.Completion(ctx, llm.CompletionParams{
    Model: "kimi-k3-highspeed",
    Messages: messages,
})

if response.Choices[0].Message.Reasoning != nil {
    fmt.Println("Thinking:", response.Choices[0].Message.Reasoning.Content)
}
```

**Images and Video:**

Images and videos must be sent as base64 data URLs — public image URLs are rejected. Large videos can be uploaded once and referenced as `ms://<file-id>`:

```go
{Role: llm.ROLE_USER, Content: llm.ContentFromParts(
    &llm.ContentPartText{Text: "Describe this video."},
    &llm.ContentPartVideo{VideoURL: &llm.VideoURL{URL: "data:video/mp4;base64,..."}},
)}
```

### OpenAI

```go
import (
    "github.com/humbornjo/llm"
    "github.com/humbornjo/llm/providers/openai"
)

// Using environment variable (OPENAI_API_KEY).
provider, err := openai.New()

// Or with explicit API key.
provider, err := openai.New(llm.WithAPIKey("sk-..."))

// Or with custom base URL (for Azure, proxies, etc.).
provider, err := openai.New(
    llm.WithAPIKey("your-key"),
    llm.WithBaseURL("https://your-endpoint.openai.azure.com"),
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

## Coming Soon

The following providers are planned for future releases:

| Provider     | Status                                            |
|--------------|---------------------------------------------------|
| Cohere       | Planned                                           |
| Together AI  | Planned                                           |
| AWS Bedrock  | Planned                                           |
| Azure OpenAI | Planned (use OpenAI with custom base URL for now) |

## Adding a New Provider

Want to add support for a new provider? The existing packages are the reference:
`providers/openai` shows a full SDK-backed implementation, and
`providers/deepseek` shows how to wrap it for an OpenAI-compatible API.

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
    ID                string   `json:"id"`
    Object            string   `json:"object"`
    Created           int64    `json:"created"`
    Model             string   `json:"model"`
    Choices           []Choice `json:"choices"`
    Usage             *Usage   `json:"usage,omitempty"`
    SystemFingerprint string   `json:"system_fingerprint,omitempty"`
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
