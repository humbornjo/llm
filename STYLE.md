# Code Style

`llm` follows the same pragmatic, package-oriented Go style as Fuss. Keep the
implementation direct, make ownership visible, and prefer a small amount of
plain code over speculative abstractions.

## Naming

- Use concise conventional locals such as `ctx`, `cfg`, `req`, `resp`, `err`,
  `opts`, and `cli` when their roles are clear.
- Keep receivers short and stable, normally `p`, `c`, or `t`.
- Use one-word lowercase package names. Do not add import aliases when the
  declared package name already reads correctly.
- Use `New` for the primary package constructor, `NewX` for other constructors,
  and `FromX` for adapters.
- Name interfaces for capabilities or behavior, such as `Provider`, `Tool`,
  `ModelLister`, and `ErrorConverter`.
- Keep exported types, functions, and methods in Go-style PascalCase. Preserve
  established initialisms such as `API`, `URL`, `JSON`, and `ID` in those
  identifiers.
- Prefer named constants for protocol values that occur in production logic.
  Do not extract a literal used once merely to avoid writing the literal.
- Name package-level primitive constants that represent protocol values,
  enumerations, defaults, or metadata in uppercase snake case. Exported values
  have no prefix, such as `ROLE_SYSTEM`, `CODE_RATE_LIMIT`, and `VERSION`.
  Package-private values have a leading underscore, such as
  `_DEFAULT_BASE_URL` and `_OBJECT_LIST`.
- Keep callable variables, maps, slices, sentinel errors, and other object
  globals in normal Go naming. Examples include `NewConfig`,
  `supportedProviders`, `ProviderModelMap`, and `ErrRateLimit`.
- Treat primitive-global renames as breaking changes. Replace the old name
  instead of retaining a compatibility alias.

## Implementation

- Keep the happy path readable from top to bottom. Handle failures early and
  avoid deep nesting.
- Prefer concrete code over wrappers, factories, or interfaces that have only
  one real implementation. Add an abstraction when it represents an external
  boundary or supports genuine alternatives.
- Do not add pass-through helpers that provide no validation, translation,
  policy, or lifecycle behavior.
- Keep a cohesive operation inline when it has one caller. Extract substantial
  protocol conversion, concurrent lifecycle, or independently testable I/O
  boundaries when doing so clarifies the caller.
- Do not mutate caller-owned parameters, maps, or slices. Clone mutable data
  before normalizing or enriching it.
- Pass `context.Context` through network and provider calls. Do not replace a
  caller context with `context.Background()` inside reusable code.
- Use `iter.Seq2[T, error]` for streams. Stop provider work when the consumer
  stops iteration, and surface `ctx.Err()` when cancellation terminates a
  stream.
- Close response bodies, streams, and goroutines on every success, error,
  cancellation, and early-consumer-exit path.
- Use standard-library functionality before introducing a dependency.

## Packages and Providers

- Keep provider-neutral contracts in the root package or `providers`; keep SDK
  translation inside the owning provider package.
- Put interfaces at real external boundaries. Use compile-time assertions when
  they make provider capabilities explicit.
- Prefer a thin wrapper around `providers/openai.CompatibleProvider` for a
  genuinely OpenAI-compatible provider.
- Validate required provider-neutral input before issuing a network request.
  Reject unknown roles, finish reasons, and enum values rather than silently
  coercing them.
- Convert typed SDK errors with `errors.As` where possible. Preserve the cause
  with `%w` and avoid matching error strings unless the remote protocol offers
  no structured alternative.
- Keep external protocol names intact. Mozilla attribution, AnyLLM Platform
  headers, and environment variables are external compatibility contracts, not
  local package branding.

## Tools

- A `Tool` owns both its declaration and execution. Keep argument decoding at
  the generic tool boundary and typed behavior inside the implementation.
- Return `ErrToolNotFound` for an unknown function name and wrap it with that
  name for diagnostics.
- Pass execution metadata through `ToolOption`; do not put transport-specific
  fields into provider-neutral tool schemas.

## Errors and Logging

- Return errors from library packages; do not panic for provider, validation,
  or transport failures.
- Use sentinel errors for stable categories and typed errors for structured
  details.
- Write lowercase error messages without terminal punctuation. Add operation
  context once and avoid repeating a provider name already rendered by the
  wrapped error.
- Never log prompts, message content, API keys, authorization headers, or raw
  provider payloads. A library should normally return errors and let its caller
  choose the logging policy.

## Tests

- Name Go tests `Test<PackageNameInCamelCase>_<Target>`, for example
  `TestLLM_NewToolsHandler`, `TestOpenAI_CompletionStream`, and
  `TestAnthropic_ConvertError`. For an external test package, omit the `_test`
  suffix from the package-name segment.
- Name tests after the behavior under test and use table-driven cases when one
  target has several inputs.
- Use `testify/require` for fatal assertions. Avoid redundant assertions.
- Call `t.Parallel()` when the test does not mutate process-wide state. Tests
  using `t.Setenv()` or shared servers must remain serial where necessary.
- Name table cases `tc`, and mark helpers with `t.Helper()`.
- Prefer protocol fixtures and fake servers to mocks of internal implementation
  details.
- Integration tests must skip when their required credentials are absent and
  must not run merely because a developer happens to have unrelated provider
  credentials loaded.
- Test stream completion, provider errors, context cancellation, and early
  consumer exit.

## Documentation and Verification

- Keep examples compilable and use the natural `llm` package name without a
  redundant import alias.
- Document public behavior and compatibility constraints rather than internal
  line-by-line implementation.
- Run `gofmt` on changed Go files, then `go test -race -short ./...`,
  `go vet ./...`, the configured linter when available, and `git diff --check`.
- Run credentialed integration tests separately and report their environment
  and provider failures independently from unit-test results.
