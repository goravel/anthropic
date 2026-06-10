# Anthropic

The Anthropic provider for `facades.AI()` of Goravel.

## Version

| goravel/anthropic | goravel/framework |
|-------------------|-------------------|
| v1.18.x           | v1.18.x           |

## Install

Run the command below in your project to install the package automatically:

```bash
./artisan package:install github.com/goravel/anthropic
```

This registers the service provider and updates `config/ai.go` so `ai.providers.anthropic.via` resolves through `anthropicfacades.Anthropic("anthropic")`.

Or check [the setup file](./setup/setup.go) to install the package manually.

## Custom Failover

The provider marks these Anthropic API errors as failoverable by default:

| Error | Reason |
|-------|--------|
| `rate_limit_error` or `429 Too Many Requests` | `rate_limited` |
| `billing_error` or `402 Payment Required` | `insufficient_credits` |
| `overloaded_error`, `503 Service Unavailable`, or `529 Overloaded` | `provider_overloaded` |

Configure `failover` rules to add Anthropic-specific error message mappings. Plain strings use substring matching, and slash-delimited strings use Go regular expressions.

```go
"anthropic": map[string]any{
	"key": config.Env("ANTHROPIC_API_KEY", ""),
	"failover": map[string][]string{
		"context_length_exceeded": {
			"maximum context length",
			"/(?i)context.*length/",
		},
	},
	"via": func() (ai.Provider, error) {
		return anthropicfacades.Anthropic("anthropic")
	},
}
```

## Supported capabilities

- Text prompting
- Streaming responses
- Tool calling
- Prompt attachments
- Provider-managed files via Anthropic's beta files API

## Not supported

- Image generation
- Audio generation
- Transcription

Anthropic's Go SDK does not expose OpenAI-style image, text-to-speech, or transcription APIs, so this driver only implements the capabilities Anthropic supports directly.

## Testing

Run command below to run all tests:

```bash
go test ./...
```

Run the live Anthropic smoke test with a real API key:

```bash
ANTHROPIC_API_KEY=your-key go test -run '^TestProviderPromptIntegration$' -v ./...
```

The smoke test skips automatically when `ANTHROPIC_API_KEY` is not set.
