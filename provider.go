package anthropic

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"path/filepath"
	"strings"

	goanthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	frameworkai "github.com/goravel/framework/ai"
	contractsai "github.com/goravel/framework/contracts/ai"
	contractsconfig "github.com/goravel/framework/contracts/config"
	"github.com/goravel/framework/errors"
)

const (
	DefaultTextModel = "claude-sonnet-4-5"
	defaultMaxTokens = 4096
)

const (
	defaultFailoverReasonRateLimited         contractsai.FailoverReason = "rate_limited"
	defaultFailoverReasonInsufficientCredits contractsai.FailoverReason = "insufficient_credits"
	defaultFailoverReasonProviderOverloaded  contractsai.FailoverReason = "provider_overloaded"
	anthropicStatusOverloaded                                           = 529
)

type Provider struct {
	client        goanthropic.Client
	config        contractsai.ProviderConfig
	failoverRules *frameworkai.FailoverRules
	name          string
}

func NewAnthropic(config contractsconfig.Config, provider string) (*Provider, error) {
	var providerConfig contractsai.ProviderConfig
	err := config.UnmarshalKey("ai.providers."+provider, &providerConfig)
	if err != nil {
		return nil, err
	}

	failoverRules, err := newFailoverRules(provider, providerConfig.Failover)
	if err != nil {
		return nil, err
	}

	opts := []option.RequestOption{
		option.WithoutEnvironmentDefaults(),
		option.WithAPIKey(providerConfig.Key),
	}
	if providerConfig.Url != "" {
		opts = append(opts, option.WithBaseURL(providerConfig.Url))
	}
	if providerConfig.Models.Text.Default == "" {
		providerConfig.Models.Text.Default = DefaultTextModel
	}
	if providerConfig.Models.Text.MaxTokens == 0 {
		providerConfig.Models.Text.MaxTokens = defaultMaxTokens
	}

	return &Provider{
		client:        goanthropic.NewClient(opts...),
		config:        providerConfig,
		failoverRules: failoverRules,
		name:          provider,
	}, nil
}

func (r *Provider) Prompt(ctx context.Context, prompt contractsai.AgentPrompt) (contractsai.AgentResponse, error) {
	params, err := r.buildRequest(ctx, prompt)
	if err != nil {
		return nil, err
	}

	message, err := r.client.Messages.New(ctx, params)
	if err != nil {
		return nil, r.failoverError(err)
	}

	text, toolCalls := r.parseMessageContent(message.Content)
	return frameworkai.NewTextResponse(text, r.parseUsage(message.Usage), toolCalls), nil
}

func (r *Provider) Stream(ctx context.Context, prompt contractsai.AgentPrompt) (contractsai.StreamableAgentResponse, error) {
	params, err := r.buildRequest(ctx, prompt)
	if err != nil {
		return nil, err
	}

	return frameworkai.NewStreamableResponse(ctx, func(streamCtx context.Context, emit func(contractsai.StreamEvent) error) (contractsai.AgentResponse, error) {
		stream := r.client.Messages.NewStreaming(streamCtx, params)
		defer errors.Ignore(stream.Close)

		text := strings.Builder{}
		currentUsage := contractsai.Usage(frameworkai.NewUsage(0, 0, 0))
		toolCallsByIndex := make(map[int64]contractsai.ToolCall)
		toolInputBuffer := make(map[int64]*strings.Builder)
		toolOrder := make([]int64, 0)
		var finalToolCalls []contractsai.ToolCall

		for stream.Next() {
			event := stream.Current()
			switch chunk := event.AsAny().(type) {
			case goanthropic.ContentBlockStartEvent:
				switch block := chunk.ContentBlock.AsAny().(type) {
				case goanthropic.ToolUseBlock:
					call := contractsai.ToolCall{
						ID:      block.ID,
						Name:    block.Name,
						RawArgs: strings.TrimSpace(string(block.Input)),
					}
					if call.RawArgs != "" {
						_ = json.Unmarshal([]byte(call.RawArgs), &call.Args)
					}
					toolCallsByIndex[chunk.Index] = call
					toolInputBuffer[chunk.Index] = &strings.Builder{}
					toolOrder = append(toolOrder, chunk.Index)
				}
			case goanthropic.ContentBlockDeltaEvent:
				switch delta := chunk.Delta.AsAny().(type) {
				case goanthropic.TextDelta:
					text.WriteString(delta.Text)
					if err := emit(contractsai.StreamEvent{Type: contractsai.StreamEventTypeTextDelta, Delta: delta.Text}); err != nil {
						return nil, err
					}
				case goanthropic.InputJSONDelta:
					if builder, ok := toolInputBuffer[chunk.Index]; ok {
						builder.WriteString(delta.PartialJSON)
					}
				}
			case goanthropic.ContentBlockStopEvent:
				call, ok := toolCallsByIndex[chunk.Index]
				if !ok {
					continue
				}
				if builder, ok := toolInputBuffer[chunk.Index]; ok {
					if rawArgs := strings.TrimSpace(builder.String()); rawArgs != "" {
						call.RawArgs = rawArgs
						_ = json.Unmarshal([]byte(call.RawArgs), &call.Args)
					}
				}
				toolCallsByIndex[chunk.Index] = call
			case goanthropic.MessageDeltaEvent:
				currentUsage = r.parseMessageDeltaUsage(chunk.Usage)
			case goanthropic.MessageStopEvent:
				finalToolCalls = orderedToolCalls(toolOrder, toolCallsByIndex)
			}
		}

		if err := stream.Err(); err != nil {
			if streamCtx.Err() == nil {
				if emitErr := emit(contractsai.StreamEvent{Type: contractsai.StreamEventTypeError, Error: err.Error()}); emitErr != nil {
					return nil, emitErr
				}
			}

			return nil, r.failoverError(err)
		}

		if len(finalToolCalls) == 0 {
			finalToolCalls = orderedToolCalls(toolOrder, toolCallsByIndex)
		}

		if err := emit(contractsai.StreamEvent{Type: contractsai.StreamEventTypeDone, Usage: currentUsage}); err != nil {
			return nil, err
		}

		return frameworkai.NewTextResponse(text.String(), currentUsage, finalToolCalls), nil
	}), nil
}

func (r *Provider) PutFile(ctx context.Context, file contractsai.StorableFile) (contractsai.FileResponse, error) {
	content, err := file.Content(ctx)
	if err != nil {
		return nil, err
	}

	upload, err := r.client.Beta.Files.Upload(ctx, goanthropic.BetaFileUploadParams{
		File: goanthropic.File(bytes.NewReader(content), r.uploadFilename(file), file.MimeType()),
	})
	if err != nil {
		return nil, r.failoverError(err)
	}

	return frameworkai.NewFileResponse(upload.ID, "", nil), nil
}

func (r *Provider) GetFile(ctx context.Context, id string) (contractsai.FileResponse, error) {
	metadata, err := r.client.Beta.Files.GetMetadata(ctx, id, goanthropic.BetaFileGetMetadataParams{})
	if err != nil {
		return nil, r.failoverError(err)
	}

	response, err := r.client.Beta.Files.Download(ctx, id, goanthropic.BetaFileDownloadParams{})
	if err != nil {
		return nil, r.failoverError(err)
	}
	defer errors.Ignore(response.Body.Close)

	content, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}

	mimeType := metadata.MimeType
	if mimeType == "" {
		mimeType = response.Header.Get("Content-Type")
	}

	return frameworkai.NewFileResponse(metadata.ID, mimeType, content), nil
}

func (r *Provider) DeleteFile(ctx context.Context, id string) error {
	_, err := r.client.Beta.Files.Delete(ctx, id, goanthropic.BetaFileDeleteParams{})
	if err != nil {
		return r.failoverError(err)
	}

	return nil
}

func (r *Provider) failoverError(err error) error {
	var anthropicErr *goanthropic.Error
	if stderrors.As(err, &anthropicErr) {
		switch anthropicErr.Type() {
		case goanthropic.ErrorTypeRateLimitError:
			return frameworkai.NewFailoverError(r.providerName(), defaultFailoverReasonRateLimited, err)
		case goanthropic.ErrorTypeBillingError:
			return frameworkai.NewFailoverError(r.providerName(), defaultFailoverReasonInsufficientCredits, err)
		case goanthropic.ErrorTypeOverloadedError:
			return frameworkai.NewFailoverError(r.providerName(), defaultFailoverReasonProviderOverloaded, err)
		}

		switch anthropicErr.StatusCode {
		case http.StatusTooManyRequests:
			return frameworkai.NewFailoverError(r.providerName(), defaultFailoverReasonRateLimited, err)
		case http.StatusPaymentRequired:
			return frameworkai.NewFailoverError(r.providerName(), defaultFailoverReasonInsufficientCredits, err)
		case http.StatusServiceUnavailable, anthropicStatusOverloaded:
			return frameworkai.NewFailoverError(r.providerName(), defaultFailoverReasonProviderOverloaded, err)
		}
	}

	if r.failoverRules == nil {
		return err
	}

	return r.failoverRules.Wrap(r.providerName(), err)
}

func (r *Provider) providerName() string {
	if r.name != "" {
		return r.name
	}

	return "anthropic"
}

func newFailoverRules(provider string, patterns map[contractsai.FailoverReason][]string) (*frameworkai.FailoverRules, error) {
	if !hasFailoverPatterns(patterns) {
		return nil, nil
	}

	rules, err := frameworkai.NewFailoverRules(provider, patterns)
	if err != nil {
		return nil, err
	}

	return &rules, nil
}

func hasFailoverPatterns(patterns map[contractsai.FailoverReason][]string) bool {
	for reason, reasonPatterns := range patterns {
		if reason == "" {
			continue
		}
		for _, pattern := range reasonPatterns {
			if pattern != "" {
				return true
			}
		}
	}

	return false
}

func (r *Provider) buildRequest(ctx context.Context, prompt contractsai.AgentPrompt) (goanthropic.MessageNewParams, error) {
	messages, err := r.buildMessages(ctx, prompt)
	if err != nil {
		return goanthropic.MessageNewParams{}, err
	}

	params := goanthropic.MessageNewParams{
		MaxTokens: int64(r.config.Models.Text.MaxTokens),
		Messages:  messages,
		Model:     goanthropic.Model(r.resolveModel(prompt.Model)),
	}
	if instructions := prompt.Agent.Instructions(); instructions != "" {
		params.System = []goanthropic.TextBlockParam{{Text: instructions}}
	}
	if len(prompt.Tools) > 0 {
		params.Tools = r.buildTools(prompt.Tools)
		params.ToolChoice = goanthropic.ToolChoiceUnionParam{
			OfAuto: &goanthropic.ToolChoiceAutoParam{},
		}
	}

	return params, nil
}

func (r *Provider) buildMessages(ctx context.Context, prompt contractsai.AgentPrompt) ([]goanthropic.MessageParam, error) {
	messages := make([]goanthropic.MessageParam, 0)
	history := prompt.Agent.Messages()
	attachmentIndex := -1
	if prompt.Input == "" && len(prompt.Attachments) > 0 {
		for i := len(history) - 1; i >= 0; i-- {
			if history[i].Role == contractsai.RoleUser {
				attachmentIndex = i
				break
			}
		}
	}

	for i, message := range history {
		switch message.Role {
		case contractsai.RoleUser:
			attachments := []contractsai.Attachment(nil)
			if i == attachmentIndex {
				attachments = prompt.Attachments
			}

			built, err := r.buildUserMessage(ctx, message.Content, attachments)
			if err != nil {
				return nil, err
			}
			messages = append(messages, built)
		case contractsai.RoleAssistant:
			built := r.buildAssistantMessage(message)
			if len(built.Content) > 0 {
				messages = append(messages, built)
			}
		case contractsai.RoleToolResult:
			messages = append(messages, r.buildToolResultMessage(message))
		}
	}

	if prompt.Input != "" || (len(prompt.Attachments) > 0 && attachmentIndex == -1) {
		built, err := r.buildUserMessage(ctx, prompt.Input, prompt.Attachments)
		if err != nil {
			return nil, err
		}
		messages = append(messages, built)
	}

	return messages, nil
}

func (r *Provider) buildUserMessage(ctx context.Context, input string, attachments []contractsai.Attachment) (goanthropic.MessageParam, error) {
	blocks := make([]goanthropic.ContentBlockParamUnion, 0, len(attachments)+1)
	if input != "" || len(attachments) == 0 {
		blocks = append(blocks, goanthropic.ContentBlockParamUnion{OfText: &goanthropic.TextBlockParam{Text: input}})
	}

	for _, attachment := range attachments {
		block, err := r.buildAttachmentBlock(ctx, attachment)
		if err != nil {
			return goanthropic.MessageParam{}, err
		}
		blocks = append(blocks, block)
	}

	return goanthropic.NewUserMessage(blocks...), nil
}

func (r *Provider) buildAssistantMessage(message contractsai.Message) goanthropic.MessageParam {
	blocks := make([]goanthropic.ContentBlockParamUnion, 0, len(message.ToolCalls)+1)
	if message.Content != "" || len(message.ToolCalls) == 0 {
		blocks = append(blocks, goanthropic.ContentBlockParamUnion{OfText: &goanthropic.TextBlockParam{Text: message.Content}})
	}
	for _, toolCall := range message.ToolCalls {
		input := map[string]any{}
		if toolCall.RawArgs != "" {
			_ = json.Unmarshal([]byte(toolCall.RawArgs), &input)
		} else if toolCall.Args != nil {
			input = toolCall.Args
		}
		blocks = append(blocks, goanthropic.ContentBlockParamUnion{OfToolUse: &goanthropic.ToolUseBlockParam{
			ID:    toolCall.ID,
			Name:  toolCall.Name,
			Input: input,
		}})
	}

	return goanthropic.NewAssistantMessage(blocks...)
}

func (r *Provider) buildToolResultMessage(message contractsai.Message) goanthropic.MessageParam {
	return goanthropic.NewUserMessage(goanthropic.ContentBlockParamUnion{OfToolResult: &goanthropic.ToolResultBlockParam{
		ToolUseID: message.ToolCallID,
		Content: []goanthropic.ToolResultBlockParamContentUnion{{
			OfText: &goanthropic.TextBlockParam{Text: message.Content},
		}},
	}})
}

func (r *Provider) buildAttachmentBlock(ctx context.Context, attachment contractsai.Attachment) (goanthropic.ContentBlockParamUnion, error) {
	content, fileName, mimeType, err := r.resolveAttachment(ctx, attachment)
	if err != nil {
		return goanthropic.ContentBlockParamUnion{}, err
	}

	switch attachment.Kind() {
	case contractsai.AttachmentKindImage:
		mediaType := normalizeMediaType(mimeType)
		if mediaType == "" {
			mediaType = "image/png"
		}

		return goanthropic.ContentBlockParamUnion{OfImage: &goanthropic.ImageBlockParam{
			Source: goanthropic.ImageBlockParamSourceUnion{OfBase64: &goanthropic.Base64ImageSourceParam{
				Data:      base64.StdEncoding.EncodeToString(content),
				MediaType: goanthropic.Base64ImageSourceMediaType(mediaType),
			}},
		}}, nil
	case contractsai.AttachmentKindFile:
		mediaType := normalizeMediaType(mimeType)
		switch {
		case strings.HasPrefix(mediaType, "text/") || mediaType == "application/json" || mediaType == "application/xml" || mediaType == "application/yaml" || mediaType == "application/x-yaml":
			return goanthropic.ContentBlockParamUnion{OfDocument: &goanthropic.DocumentBlockParam{
				Title: goanthropic.String(fileName),
				Source: goanthropic.DocumentBlockParamSourceUnion{OfText: &goanthropic.PlainTextSourceParam{
					Data: string(content),
				}},
			}}, nil
		case mediaType == "application/pdf":
			return goanthropic.ContentBlockParamUnion{OfDocument: &goanthropic.DocumentBlockParam{
				Title: goanthropic.String(fileName),
				Source: goanthropic.DocumentBlockParamSourceUnion{OfBase64: &goanthropic.Base64PDFSourceParam{
					Data: base64.StdEncoding.EncodeToString(content),
				}},
			}}, nil
		default:
			return goanthropic.ContentBlockParamUnion{}, errors.AIUnsupportedAttachmentKind.Args(attachment.Kind())
		}
	default:
		return goanthropic.ContentBlockParamUnion{}, errors.AIUnsupportedAttachmentKind.Args(attachment.Kind())
	}
}

func (r *Provider) resolveAttachment(ctx context.Context, attachment contractsai.Attachment) ([]byte, string, string, error) {
	if stored, ok := attachment.(contractsai.ProviderFile); ok && stored.ID() != "" {
		file, err := r.GetFile(ctx, stored.ID())
		if err != nil {
			return nil, "", "", err
		}
		content, err := file.Content(ctx)
		if err != nil {
			return nil, "", "", err
		}

		return content, attachment.FileName(), file.MimeType(), nil
	}

	content, err := attachment.Content(ctx)
	if err != nil {
		return nil, "", "", err
	}

	return content, attachment.FileName(), attachment.MimeType(), nil
}

func (r *Provider) buildTools(tools []contractsai.Tool) []goanthropic.ToolUnionParam {
	params := make([]goanthropic.ToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		inputSchema := goanthropic.ToolInputSchemaParam{Type: "object"}
		if schema := tool.Parameters(); schema != nil {
			inputSchema.Properties = schema["properties"]
			if required, ok := schema["required"].([]string); ok {
				inputSchema.Required = required
			} else if requiredAny, ok := schema["required"].([]any); ok {
				inputSchema.Required = toStringSlice(requiredAny)
			}
			inputSchema.ExtraFields = cloneSchemaExtras(schema)
		}

		param := goanthropic.ToolParam{
			Name:        tool.Name(),
			InputSchema: inputSchema,
			Type:        goanthropic.ToolTypeCustom,
			Strict:      goanthropic.Bool(true),
		}
		if description := tool.Description(); description != "" {
			param.Description = goanthropic.String(description)
		}

		params = append(params, goanthropic.ToolUnionParam{OfTool: &param})
	}

	return params
}

func (r *Provider) parseMessageContent(content []goanthropic.ContentBlockUnion) (string, []contractsai.ToolCall) {
	text := strings.Builder{}
	toolCalls := make([]contractsai.ToolCall, 0)
	for _, block := range content {
		switch value := block.AsAny().(type) {
		case goanthropic.TextBlock:
			text.WriteString(value.Text)
		case goanthropic.ToolUseBlock:
			call := contractsai.ToolCall{
				ID:      value.ID,
				Name:    value.Name,
				RawArgs: strings.TrimSpace(string(value.Input)),
			}
			if call.RawArgs != "" {
				_ = json.Unmarshal([]byte(call.RawArgs), &call.Args)
			}
			toolCalls = append(toolCalls, call)
		}
	}
	if len(toolCalls) == 0 {
		return text.String(), nil
	}

	return text.String(), toolCalls
}

func (r *Provider) parseUsage(raw goanthropic.Usage) contractsai.Usage {
	input := raw.InputTokens + raw.CacheCreationInputTokens + raw.CacheReadInputTokens
	output := raw.OutputTokens
	return frameworkai.NewUsage(int(input), int(output), int(input+output))
}

func (r *Provider) parseMessageDeltaUsage(raw goanthropic.MessageDeltaUsage) contractsai.Usage {
	input := raw.InputTokens + raw.CacheCreationInputTokens + raw.CacheReadInputTokens
	output := raw.OutputTokens
	return frameworkai.NewUsage(int(input), int(output), int(input+output))
}

func (r *Provider) resolveModel(model string) string {
	if model != "" {
		return model
	}

	return r.config.Models.Text.Default
}

func (r *Provider) uploadFilename(file contractsai.StorableFile) string {
	if fileName := file.FileName(); fileName != "" {
		return fileName
	}

	mediaType := normalizeMediaType(file.MimeType())
	if extensions, err := mime.ExtensionsByType(mediaType); err == nil && len(extensions) > 0 {
		return "attachment" + extensions[0]
	}

	return fmt.Sprintf("attachment%s", fallbackFileExtension(mediaType))
}

func fallbackFileExtension(mimeType string) string {
	switch strings.ToLower(mimeType) {
	case "text/plain", "text/plain; charset=utf-8":
		return ".txt"
	case "application/json":
		return ".json"
	case "application/pdf":
		return ".pdf"
	case "image/png":
		return ".png"
	case "image/jpeg":
		return ".jpg"
	default:
		return filepath.Ext(".bin")
	}
}

func normalizeMediaType(mimeType string) string {
	if mimeType == "" {
		return ""
	}
	if parsed, _, err := mime.ParseMediaType(mimeType); err == nil {
		return parsed
	}

	return mimeType
}

func orderedToolCalls(order []int64, calls map[int64]contractsai.ToolCall) []contractsai.ToolCall {
	if len(order) == 0 {
		return nil
	}

	result := make([]contractsai.ToolCall, 0, len(order))
	for _, index := range order {
		call, ok := calls[index]
		if !ok {
			continue
		}
		result = append(result, call)
	}

	if len(result) == 0 {
		return nil
	}

	return result
}

func toStringSlice(values []any) []string {
	result := make([]string, 0, len(values))
	for _, value := range values {
		if stringValue, ok := value.(string); ok {
			result = append(result, stringValue)
		}
	}
	return result
}

func cloneSchemaExtras(schema map[string]any) map[string]any {
	if len(schema) == 0 {
		return nil
	}

	cloned := make(map[string]any, len(schema))
	for key, value := range schema {
		if key == "properties" || key == "required" || key == "type" {
			continue
		}
		cloned[key] = value
	}
	if len(cloned) == 0 {
		return nil
	}

	return cloned
}
