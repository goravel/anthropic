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
