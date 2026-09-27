// Package llm provides a unified interface for interacting with LLM providers.
//
// This package re-exports common types and configuration options from subpackages,
// allowing most use cases to work with just two imports:
//
//	import (
//	    "github.com/humbornjo/llm"
//	    "github.com/humbornjo/llm/providers/openai"
//	)
//
//	provider, err := openai.New(llm.WithAPIKey("sk-..."))
//	response, err := provider.Completion(ctx, llm.CompletionParams{
//	    Model: "gpt-4o-mini",
//	    Messages: []llm.Message{
//	        {Role: llm.ROLE_USER, Content: "Hello!"},
//	    },
//	})
package llm

import (
	"github.com/humbornjo/llm/config"
	"github.com/humbornjo/llm/errors"
	"github.com/humbornjo/llm/providers"
)

// Message roles.
const (
	ROLE_ASSISTANT = providers.ROLE_ASSISTANT
	ROLE_SYSTEM    = providers.ROLE_SYSTEM
	ROLE_TOOL      = providers.ROLE_TOOL
	ROLE_USER      = providers.ROLE_USER
)

// Content part type constants.
const (
	CONTENT_PART_TEXT        = providers.CONTENT_PART_TEXT
	CONTENT_PART_FILE        = providers.CONTENT_PART_FILE
	CONTENT_PART_IMAGE_URL   = providers.CONTENT_PART_IMAGE_URL
	CONTENT_PART_INPUT_AUDIO = providers.CONTENT_PART_INPUT_AUDIO
)

// Finish reasons.
const (
	FINISH_REASON_CONTENT_FILTER = providers.FINISH_REASON_CONTENT_FILTER
	FINISH_REASON_LENGTH         = providers.FINISH_REASON_LENGTH
	FINISH_REASON_STOP           = providers.FINISH_REASON_STOP
	FINISH_REASON_TOOL_CALLS     = providers.FINISH_REASON_TOOL_CALLS
)

// Batch status constants.
const (
	BATCH_STATUS_CANCELLED   = providers.BATCH_STATUS_CANCELLED
	BATCH_STATUS_CANCELLING  = providers.BATCH_STATUS_CANCELLING
	BATCH_STATUS_COMPLETED   = providers.BATCH_STATUS_COMPLETED
	BATCH_STATUS_EXPIRED     = providers.BATCH_STATUS_EXPIRED
	BATCH_STATUS_FAILED      = providers.BATCH_STATUS_FAILED
	BATCH_STATUS_FINALIZING  = providers.BATCH_STATUS_FINALIZING
	BATCH_STATUS_IN_PROGRESS = providers.BATCH_STATUS_IN_PROGRESS
	BATCH_STATUS_VALIDATING  = providers.BATCH_STATUS_VALIDATING
)

// ReasoningEffort levels.
const (
	REASONING_EFFORT_AUTO   = providers.REASONING_EFFORT_AUTO
	REASONING_EFFORT_HIGH   = providers.REASONING_EFFORT_HIGH
	REASONING_EFFORT_LOW    = providers.REASONING_EFFORT_LOW
	REASONING_EFFORT_MEDIUM = providers.REASONING_EFFORT_MEDIUM
	REASONING_EFFORT_NONE   = providers.REASONING_EFFORT_NONE
)

// Provider types.
type (
	BatchProvider      = providers.BatchProvider
	Capabilities       = providers.Capabilities
	CapabilityProvider = providers.CapabilityProvider
	EmbeddingProvider  = providers.EmbeddingProvider
	ModelLister        = providers.ModelLister
	ModerationProvider = providers.ModerationProvider
	Provider           = providers.Provider
	RerankProvider     = providers.RerankProvider
)

// Batch types.
type (
	Batch              = providers.Batch
	BatchRequestCounts = providers.BatchRequestCounts
	BatchRequestItem   = providers.BatchRequestItem
	BatchResult        = providers.BatchResult
	BatchResultError   = providers.BatchResultError
	BatchResultItem    = providers.BatchResultItem
	CreateBatchParams  = providers.CreateBatchParams
	ListBatchesOptions = providers.ListBatchesOptions
)

// Request/Response types.
type (
	ChatCompletion      = providers.ChatCompletion
	ChatCompletionChunk = providers.ChatCompletionChunk
	Choice              = providers.Choice
	ChunkChoice         = providers.ChunkChoice
	ChunkDelta          = providers.ChunkDelta
	CompletionParams    = providers.CompletionParams
	EmbeddingParams     = providers.EmbeddingParams
	EmbeddingResponse   = providers.EmbeddingResponse
	ModelsResponse      = providers.ModelsResponse
	ModerationParams    = providers.ModerationParams
	ModerationResponse  = providers.ModerationResponse
	ModerationResult    = providers.ModerationResult
)

// Message types.
type (
	Content          = providers.Content
	ContentString    = providers.ContentString
	ContentParts     = providers.ContentParts
	ContentPartAudio = providers.ContentPartAudio
	ContentPartFile  = providers.ContentPartFile
	ContentPartImage = providers.ContentPartImage
	ContentPartText  = providers.ContentPartText
	ContentPart      = providers.ContentPart
	ContentPartType  = providers.ContentPartType
	ImageURL         = providers.ImageURL
	InputAudio       = providers.InputAudio
	File             = providers.File
	Message          = providers.Message
	Reasoning        = providers.Reasoning
)

// Tool types.
type (
	Function           = providers.Function
	FunctionCall       = providers.FunctionCall
	ToolInfo           = providers.ToolInfo
	ToolCall           = providers.ToolCall
	ToolChoice         = providers.ToolChoice
	ToolChoiceFunction = providers.ToolChoiceFunction
)

// Response format types.
type (
	JSONSchema     = providers.JSONSchema
	ResponseFormat = providers.ResponseFormat
	StreamOptions  = providers.StreamOptions
)

// Rerank types.
type (
	RerankMeta     = providers.RerankMeta
	RerankParams   = providers.RerankParams
	RerankResponse = providers.RerankResponse
	RerankResult   = providers.RerankResult
	RerankUsage    = providers.RerankUsage
)

// Usage and model types.
type (
	BatchStatus             = providers.BatchStatus
	CompletionTokensDetails = providers.CompletionTokensDetails
	EmbeddingData           = providers.EmbeddingData
	EmbeddingUsage          = providers.EmbeddingUsage
	Model                   = providers.Model
	PromptTokensDetails     = providers.PromptTokensDetails
	ReasoningEffort         = providers.ReasoningEffort
	Usage                   = providers.Usage
)

// Config types.
type (
	Config = config.Config
	Option = config.Option
)

// Configuration options.
var (
	NewConfig         = config.New
	WithAPIKey        = config.WithAPIKey
	WithBaseURL       = config.WithBaseURL
	WithExtra         = config.WithExtra
	WithHTTPClient    = config.WithHTTPClient
	WithTimeout       = config.WithTimeout
	ContentFromParts  = providers.ContentFromParts
	ContentFromString = providers.ContentFromString
)

// Sentinel errors for type checking with errors.Is().
var (
	ErrAuthentication      = errors.ErrAuthentication
	ErrContentFilter       = errors.ErrContentFilter
	ErrContextLength       = errors.ErrContextLength
	ErrInsufficientFunds   = errors.ErrInsufficientFunds
	ErrInvalidRequest      = errors.ErrInvalidRequest
	ErrMissingAPIKey       = errors.ErrMissingAPIKey
	ErrModelNotFound       = errors.ErrModelNotFound
	ErrProvider            = errors.ErrProvider
	ErrRateLimit           = errors.ErrRateLimit
	ErrUnsupported         = errors.ErrUnsupported
	ErrUnsupportedParam    = errors.ErrUnsupportedParam
	ErrUnsupportedProvider = errors.ErrUnsupportedProvider
)

// Error types.
type (
	AuthenticationError       = errors.AuthenticationError
	BaseError                 = errors.BaseError
	ContentFilterError        = errors.ContentFilterError
	ContextLengthError        = errors.ContextLengthError
	InsufficientFundsError    = errors.InsufficientFundsError
	InvalidRequestError       = errors.InvalidRequestError
	MissingAPIKeyError        = errors.MissingAPIKeyError
	ModelNotFoundError        = errors.ModelNotFoundError
	ProviderError             = errors.ProviderError
	RateLimitError            = errors.RateLimitError
	UnsupportedOperationError = errors.UnsupportedOperationError
	UnsupportedParamError     = errors.UnsupportedParamError
	UnsupportedProviderError  = errors.UnsupportedProviderError
)
