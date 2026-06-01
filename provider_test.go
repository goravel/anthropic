package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	goanthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	frameworkai "github.com/goravel/framework/ai"
	contractsai "github.com/goravel/framework/contracts/ai"
	mocksai "github.com/goravel/framework/mocks/ai"
	mocksconfig "github.com/goravel/framework/mocks/config"
)

type capturedRequest struct {
	path        string
	apiKey      string
	version     string
	beta        string
	body        map[string]any
	contentType string
	rawBody     []byte
}

type namedAttachment struct {
	kind     contractsai.AttachmentKind
	filename string
	mimeType string
	content  []byte
}

func (attachment namedAttachment) Kind() contractsai.AttachmentKind { return attachment.kind }
func (attachment namedAttachment) FileName() string                 { return attachment.filename }
func (attachment namedAttachment) MimeType() string                 { return attachment.mimeType }
func (attachment namedAttachment) Content(context.Context) ([]byte, error) {
	return attachment.content, nil
}
func (attachment namedAttachment) Put(context.Context, ...contractsai.Option) (contractsai.FileResponse, error) {
	return nil, nil
}

type storedAttachment struct {
	namedAttachment
	id string
}

func (attachment storedAttachment) ID() string { return attachment.id }
func (attachment storedAttachment) Get(context.Context, ...contractsai.Option) (contractsai.FileResponse, error) {
	return nil, nil
}
func (attachment storedAttachment) Delete(context.Context, ...contractsai.Option) error { return nil }

type staticTool struct {
	name        string
	description string
	params      map[string]any
}

func (t *staticTool) Name() string               { return t.name }
func (t *staticTool) Description() string        { return t.description }
func (t *staticTool) Parameters() map[string]any { return t.params }
func (t *staticTool) Execute(_ context.Context, _ map[string]any) (string, error) {
	return "tool result", nil
}

func TestNewAnthropic(t *testing.T) {
	var mockConfig *mocksconfig.Config

	beforeEach := func() {
		mockConfig = mocksconfig.NewConfig(t)
	}

	tests := []struct {
		name         string
		setup        func()
		expectConfig *contractsai.ProviderConfig
		expectErr    error
	}{
		{
			name: "returns unmarshal error",
			setup: func() {
				mockConfig.EXPECT().UnmarshalKey("ai.providers.anthropic", new(contractsai.ProviderConfig)).Return(assert.AnError).Once()
			},
			expectErr: assert.AnError,
		},
		{
			name: "sets default text model",
			setup: func() {
				mockConfig.EXPECT().UnmarshalKey("ai.providers.anthropic", new(contractsai.ProviderConfig)).RunAndReturn(func(_ string, rawVal any) error {
					cfg := rawVal.(*contractsai.ProviderConfig)
					cfg.Key = "test-key"
					cfg.Url = "http://localhost:1234"
					return nil
				}).Once()
			},
			expectConfig: func() *contractsai.ProviderConfig {
				cfg := contractsai.ProviderConfig{Key: "test-key", Url: "http://localhost:1234"}
				cfg.Models.Text.Default = DefaultTextModel
				return &cfg
			}(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			beforeEach()
			tt.setup()

			provider, err := NewAnthropic(mockConfig, "anthropic")

			assert.Equal(t, tt.expectErr, err)
			if tt.expectErr != nil {
				assert.Nil(t, provider)
				return
			}
			require.NotNil(t, provider)
			assert.Equal(t, *tt.expectConfig, provider.config)
		})
	}
}

func TestProviderPrompt(t *testing.T) {
	var mockAgent *mocksai.Agent
	beforeEach := func() {
		mockAgent = mocksai.NewAgent(t)
	}

	tests := []struct {
		name        string
		body        string
		setup       func()
		input       string
		attachments []contractsai.Attachment
		tools       []contractsai.Tool
		expectText  string
		expectTools []contractsai.ToolCall
		assertBody  func(t *testing.T, body map[string]any)
	}{
		{
			name: "builds request with text history and tool definitions",
			body: anthropicMessageResponse(t, []map[string]any{{"type": "text", "text": "assistant reply"}}, 11, 7),
			setup: func() {
				mockAgent.EXPECT().Instructions().Return("system rule").Once()
				mockAgent.EXPECT().Messages().Return([]contractsai.Message{{Role: contractsai.RoleUser, Content: "history user"}, {Role: contractsai.RoleAssistant, Content: "history assistant"}}).Once()
			},
			input:      "new input",
			tools:      []contractsai.Tool{&staticTool{name: "get_weather", description: "Get weather", params: map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "required": []string{"city"}}}},
			expectText: "assistant reply",
			assertBody: func(t *testing.T, body map[string]any) {
				assert.Equal(t, "claude-default", body["model"])
				assert.Equal(t, float64(4096), body["max_tokens"])
				assert.Equal(t, []any{map[string]any{"text": "system rule", "type": "text"}}, body["system"])
				messages := body["messages"].([]any)
				require.Len(t, messages, 3)
				assert.Equal(t, "user", messages[0].(map[string]any)["role"])
				assert.Equal(t, "assistant", messages[1].(map[string]any)["role"])
				assert.Equal(t, "user", messages[2].(map[string]any)["role"])
				tools := body["tools"].([]any)
				require.Len(t, tools, 1)
				tool := tools[0].(map[string]any)
				assert.Equal(t, "get_weather", tool["name"])
				assert.Equal(t, "Get weather", tool["description"])
				assert.Equal(t, true, tool["strict"])
				assert.Equal(t, "custom", tool["type"])
			},
		},
		{
			name: "serializes attachments and returns tool calls",
			body: anthropicMessageResponse(t, []map[string]any{{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": map[string]any{"city": "London"}}}, 5, 3),
			setup: func() {
				mockAgent.EXPECT().Instructions().Return("").Once()
				mockAgent.EXPECT().Messages().Return(nil).Once()
			},
			attachments: []contractsai.Attachment{
				frameworkai.ImageFromByte([]byte("image"), frameworkai.WithMimeType("image/png")),
				namedAttachment{kind: contractsai.AttachmentKindFile, filename: "report.txt", mimeType: "text/plain", content: []byte("document")},
				namedAttachment{kind: contractsai.AttachmentKindFile, filename: "spec.pdf", mimeType: "application/pdf", content: []byte("%PDF")},
			},
			expectTools: []contractsai.ToolCall{{ID: "toolu_1", Name: "get_weather", Args: map[string]any{"city": "London"}, RawArgs: `{"city":"London"}`}},
			assertBody: func(t *testing.T, body map[string]any) {
				messages := body["messages"].([]any)
				require.Len(t, messages, 1)
				content := messages[0].(map[string]any)["content"].([]any)
				require.Len(t, content, 3)
				assert.Equal(t, "image", content[0].(map[string]any)["type"])
				assert.Equal(t, "document", content[1].(map[string]any)["type"])
				assert.Equal(t, "document", content[2].(map[string]any)["type"])
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			beforeEach()
			tt.setup()

			captured := make(chan capturedRequest, 1)
			server := newMessageServer(t, http.StatusOK, tt.body, captured)
			defer server.Close()

			provider := &Provider{
				client: goanthropic.NewClient(option.WithoutEnvironmentDefaults(), option.WithBaseURL(server.URL), option.WithAPIKey("test-key")),
				config: contractsai.ProviderConfig{},
			}
			provider.config.Models.Text.Default = "claude-default"

			response, err := provider.Prompt(context.Background(), contractsai.AgentPrompt{Agent: mockAgent, Input: tt.input, Attachments: tt.attachments, Tools: tt.tools})
			require.NoError(t, err)
			require.NotNil(t, response)
			assert.Equal(t, tt.expectText, response.Text())
			assert.Equal(t, tt.expectTools, response.ToolCalls())

			req := <-captured
			assert.Equal(t, "/v1/messages", req.path)
			assert.Equal(t, "test-key", req.apiKey)
			assert.Equal(t, "2023-06-01", req.version)
			tt.assertBody(t, req.body)
		})
	}
}

func TestProviderStream(t *testing.T) {
	mockAgent := mocksai.NewAgent(t)
	mockAgent.EXPECT().Instructions().Return("system rule").Once()
	mockAgent.EXPECT().Messages().Return(nil).Once()

	captured := make(chan capturedRequest, 1)
	server := newStreamingMessageServer(t, strings.Join([]string{
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hel\"}}\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":\"\",\"container\":{},\"stop_details\":{}},\"usage\":{\"input_tokens\":4,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0,\"output_tokens\":2,\"output_tokens_details\":{\"thinking_tokens\":0},\"server_tool_use\":{\"web_fetch_requests\":0,\"web_search_requests\":0}}}\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n",
	}, "\n"), captured)
	defer server.Close()

	provider := &Provider{
		client: goanthropic.NewClient(option.WithoutEnvironmentDefaults(), option.WithBaseURL(server.URL), option.WithAPIKey("test-key")),
		config: contractsai.ProviderConfig{},
	}
	provider.config.Models.Text.Default = "claude-default"

	stream, err := provider.Stream(context.Background(), contractsai.AgentPrompt{Agent: mockAgent, Input: "hello"})
	require.NoError(t, err)

	var events []contractsai.StreamEvent
	var response contractsai.AgentResponse
	stream.Then(func(res contractsai.AgentResponse) {
		response = res
	})
	err = stream.Each(func(event contractsai.StreamEvent) error {
		events = append(events, event)
		return nil
	})
	require.NoError(t, err)
	require.NotNil(t, response)
	assert.Equal(t, "hello", response.Text())
	require.Len(t, events, 3)
	assert.Equal(t, contractsai.StreamEventTypeTextDelta, events[0].Type)
	assert.Equal(t, "hel", events[0].Delta)
	assert.Equal(t, contractsai.StreamEventTypeTextDelta, events[1].Type)
	assert.Equal(t, "lo", events[1].Delta)
	assert.Equal(t, contractsai.StreamEventTypeDone, events[2].Type)
	assert.Equal(t, 6, events[2].Usage.Total())

	req := <-captured
	assert.Equal(t, true, req.body["stream"])
}

func TestProviderFileLifecycle(t *testing.T) {
	t.Run("put file uploads to beta files api", func(t *testing.T) {
		captured := make(chan capturedRequest, 1)
		server := newUploadServer(t, `{"id":"file_123","created_at":"2026-01-01T00:00:00Z","filename":"report.txt","mime_type":"text/plain","size_bytes":6,"type":"file","downloadable":true,"scope":null}`, captured)
		defer server.Close()

		provider := &Provider{client: goanthropic.NewClient(option.WithoutEnvironmentDefaults(), option.WithBaseURL(server.URL), option.WithAPIKey("test-key"))}
		response, err := provider.PutFile(context.Background(), namedAttachment{kind: contractsai.AttachmentKindFile, filename: "report.txt", mimeType: "text/plain", content: []byte("report")})
		require.NoError(t, err)
		assert.Equal(t, "file_123", response.ID())

		req := <-captured
		assert.Equal(t, "/v1/files", req.path)
		assert.Contains(t, req.beta, "files-api-2025-04-14")
		assert.Contains(t, req.contentType, "multipart/form-data")
		filename, mimeType, body := parseMultipartFile(t, req.contentType, req.rawBody)
		assert.Equal(t, "report.txt", filename)
		assert.Equal(t, "text/plain", mimeType)
		assert.Equal(t, []byte("report"), body)
	})

	t.Run("get file downloads metadata and content", func(t *testing.T) {
		captured := make(chan capturedRequest, 2)
		server := newFileServer(t, captured)
		defer server.Close()

		provider := &Provider{client: goanthropic.NewClient(option.WithoutEnvironmentDefaults(), option.WithBaseURL(server.URL), option.WithAPIKey("test-key"))}
		file, err := provider.GetFile(context.Background(), "file_123")
		require.NoError(t, err)
		content, err := file.Content(context.Background())
		require.NoError(t, err)
		assert.Equal(t, []byte("report"), content)
		assert.Equal(t, "text/plain", file.MimeType())
	})

	t.Run("delete file uses beta files api", func(t *testing.T) {
		captured := make(chan capturedRequest, 1)
		server := newDeleteServer(t, captured)
		defer server.Close()

		provider := &Provider{client: goanthropic.NewClient(option.WithoutEnvironmentDefaults(), option.WithBaseURL(server.URL), option.WithAPIKey("test-key"))}
		require.NoError(t, provider.DeleteFile(context.Background(), "file_123"))

		req := <-captured
		assert.Equal(t, "/v1/files/file_123", req.path)
	})
}

func TestProviderPromptResolvesStoredAttachmentByFileID(t *testing.T) {
	mockAgent := mocksai.NewAgent(t)
	mockAgent.EXPECT().Instructions().Return("").Once()
	mockAgent.EXPECT().Messages().Return(nil).Once()

	captured := make(chan capturedRequest, 3)
	server := newStoredAttachmentServer(t, captured)
	defer server.Close()

	provider := &Provider{
		client: goanthropic.NewClient(option.WithoutEnvironmentDefaults(), option.WithBaseURL(server.URL), option.WithAPIKey("test-key")),
		config: contractsai.ProviderConfig{},
	}
	provider.config.Models.Text.Default = "claude-default"

	response, err := provider.Prompt(context.Background(), contractsai.AgentPrompt{Agent: mockAgent, Attachments: []contractsai.Attachment{storedAttachment{namedAttachment: namedAttachment{kind: contractsai.AttachmentKindFile}, id: "file_123"}}})
	require.NoError(t, err)
	assert.Equal(t, "ok", response.Text())

	requests := drainCapturedRequests(captured)
	require.Len(t, requests, 3)
	assert.Equal(t, "/v1/files/file_123", requests[0].path)
	assert.Equal(t, "/v1/files/file_123/content", requests[1].path)
	assert.Equal(t, "/v1/messages", requests[2].path)
}

func newMessageServer(t *testing.T, status int, body string, captured chan<- capturedRequest) *httptest.Server {
	t.Helper()
	handler := func(w http.ResponseWriter, r *http.Request) {
		defer func() { _ = r.Body.Close() }()
		payload, raw := decodeBodyMap(t, r)
		captured <- capturedRequest{path: r.URL.Path, apiKey: r.Header.Get("X-Api-Key"), version: r.Header.Get("anthropic-version"), body: payload, contentType: r.Header.Get("Content-Type"), rawBody: raw}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}
	return httptest.NewServer(http.HandlerFunc(handler))
}

func newStreamingMessageServer(t *testing.T, body string, captured chan<- capturedRequest) *httptest.Server {
	t.Helper()
	handler := func(w http.ResponseWriter, r *http.Request) {
		defer func() { _ = r.Body.Close() }()
		payload, raw := decodeBodyMap(t, r)
		captured <- capturedRequest{path: r.URL.Path, apiKey: r.Header.Get("X-Api-Key"), version: r.Header.Get("anthropic-version"), body: payload, contentType: r.Header.Get("Content-Type"), rawBody: raw}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(body))
	}
	return httptest.NewServer(http.HandlerFunc(handler))
}

func newUploadServer(t *testing.T, body string, captured chan<- capturedRequest) *httptest.Server {
	t.Helper()
	handler := func(w http.ResponseWriter, r *http.Request) {
		defer func() { _ = r.Body.Close() }()
		raw, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		captured <- capturedRequest{path: r.URL.Path, apiKey: r.Header.Get("X-Api-Key"), version: r.Header.Get("anthropic-version"), beta: r.Header.Get("anthropic-beta"), contentType: r.Header.Get("Content-Type"), rawBody: raw}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(body))
	}
	return httptest.NewServer(http.HandlerFunc(handler))
}

func newFileServer(t *testing.T, captured chan<- capturedRequest) *httptest.Server {
	t.Helper()
	handler := func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/files/file_123":
			captured <- capturedRequest{path: r.URL.Path, beta: r.Header.Get("anthropic-beta")}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"file_123","created_at":"2026-01-01T00:00:00Z","filename":"report.txt","mime_type":"text/plain","size_bytes":6,"type":"file","downloadable":true,"scope":null}`))
		case "/v1/files/file_123/content":
			captured <- capturedRequest{path: r.URL.Path, beta: r.Header.Get("anthropic-beta")}
			w.Header().Set("Content-Type", "application/binary")
			_, _ = w.Write([]byte("report"))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}
	return httptest.NewServer(http.HandlerFunc(handler))
}

func newDeleteServer(t *testing.T, captured chan<- capturedRequest) *httptest.Server {
	t.Helper()
	handler := func(w http.ResponseWriter, r *http.Request) {
		captured <- capturedRequest{path: r.URL.Path, beta: r.Header.Get("anthropic-beta")}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"file_123","type":"file_deleted"}`))
	}
	return httptest.NewServer(http.HandlerFunc(handler))
}

func newStoredAttachmentServer(t *testing.T, captured chan<- capturedRequest) *httptest.Server {
	t.Helper()
	handler := func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/files/file_123":
			captured <- capturedRequest{path: r.URL.Path}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"file_123","created_at":"2026-01-01T00:00:00Z","filename":"report.txt","mime_type":"text/plain","size_bytes":6,"type":"file","downloadable":true,"scope":null}`))
		case "/v1/files/file_123/content":
			captured <- capturedRequest{path: r.URL.Path}
			w.Header().Set("Content-Type", "application/binary")
			_, _ = w.Write([]byte("report"))
		case "/v1/messages":
			payload, raw := decodeBodyMap(t, r)
			captured <- capturedRequest{path: r.URL.Path, body: payload, rawBody: raw}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(anthropicMessageResponse(t, []map[string]any{{"type": "text", "text": "ok"}}, 1, 1)))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}
	return httptest.NewServer(http.HandlerFunc(handler))
}

func decodeBodyMap(t *testing.T, r *http.Request) (map[string]any, []byte) {
	t.Helper()
	body, err := io.ReadAll(r.Body)
	require.NoError(t, err)
	r.Body = io.NopCloser(strings.NewReader(string(body)))
	var payload map[string]any
	require.NoErrorf(t, json.Unmarshal(body, &payload), "failed to unmarshal request body: %s", string(body))
	return payload, body
}

func parseMultipartFile(t *testing.T, contentType string, raw []byte) (string, string, []byte) {
	t.Helper()
	_, params, err := mime.ParseMediaType(contentType)
	require.NoError(t, err)
	reader := multipart.NewReader(strings.NewReader(string(raw)), params["boundary"])
	part, err := reader.NextPart()
	require.NoError(t, err)
	defer func() { require.NoError(t, part.Close()) }()
	body, err := io.ReadAll(part)
	require.NoError(t, err)
	return part.FileName(), part.Header.Get("Content-Type"), body
}

func drainCapturedRequests(ch <-chan capturedRequest) []capturedRequest {
	var requests []capturedRequest
	for {
		select {
		case req := <-ch:
			requests = append(requests, req)
		default:
			return requests
		}
	}
}

func anthropicMessageResponse(t *testing.T, content []map[string]any, inputTokens, outputTokens int) string {
	t.Helper()
	payload := map[string]any{
		"id":      "msg_123",
		"type":    "message",
		"role":    "assistant",
		"model":   "claude-sonnet-4-5",
		"content": content,
		"stop_reason": func() any {
			if len(content) > 0 && content[0]["type"] == "tool_use" {
				return "tool_use"
			}
			return "end_turn"
		}(),
		"stop_sequence": "",
		"usage": map[string]any{
			"input_tokens":                inputTokens,
			"cache_creation_input_tokens": 0,
			"cache_read_input_tokens":     0,
			"output_tokens":               outputTokens,
			"output_tokens_details": map[string]any{
				"thinking_tokens": 0,
			},
			"server_tool_use": map[string]any{
				"web_fetch_requests":  0,
				"web_search_requests": 0,
			},
			"service_tier": "standard",
			"cache_creation": map[string]any{
				"ephemeral_1h_input_tokens": 0,
				"ephemeral_5m_input_tokens": 0,
			},
			"inference_geo": "",
		},
		"container":    map[string]any{},
		"stop_details": map[string]any{},
	}
	encoded, err := json.Marshal(payload)
	require.NoError(t, err)
	return string(encoded)
}
