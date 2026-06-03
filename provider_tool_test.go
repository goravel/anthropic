package anthropic

import (
	"context"
	"strings"
	"testing"

	goanthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	contractsai "github.com/goravel/framework/contracts/ai"
	mocksai "github.com/goravel/framework/mocks/ai"
)

func TestProviderPromptWithTools(t *testing.T) {
	mockAgent := mocksai.NewAgent(t)
	mockAgent.EXPECT().Instructions().Return("").Once()
	mockAgent.EXPECT().Messages().Return(nil).Once()

	captured := make(chan capturedRequest, 1)
	server := newMessageServer(t, httpStatusOK, anthropicMessageResponse(t, []map[string]any{{
		"type":  "tool_use",
		"id":    "toolu_1",
		"name":  "get_weather",
		"input": map[string]any{"city": "London"},
	}}, 5, 3), captured)
	defer server.Close()

	provider := &Provider{
		client: goanthropic.NewClient(option.WithoutEnvironmentDefaults(), option.WithBaseURL(server.URL), option.WithAPIKey("test-key")),
		config: contractsai.ProviderConfig{},
	}
	provider.config.Models.Text.Default = "claude-default"
	provider.config.Models.Text.MaxTokens = defaultMaxTokens

	response, err := provider.Prompt(context.Background(), contractsai.AgentPrompt{
		Agent: mockAgent,
		Input: "What is the weather in London?",
		Tools: []contractsai.Tool{&staticTool{name: "get_weather", description: "Get weather", params: map[string]any{"type": "object"}}},
	})
	require.NoError(t, err)
	require.Len(t, response.ToolCalls(), 1)
	assert.Equal(t, contractsai.ToolCall{ID: "toolu_1", Name: "get_weather", Args: map[string]any{"city": "London"}, RawArgs: `{"city":"London"}`}, response.ToolCalls()[0])

	req := <-captured
	tools := req.body["tools"].([]any)
	require.Len(t, tools, 1)
	assert.Equal(t, "get_weather", tools[0].(map[string]any)["name"])
	assert.Equal(t, "custom", tools[0].(map[string]any)["type"])
}

func TestProviderPromptBuildsToolHistory(t *testing.T) {
	mockAgent := mocksai.NewAgent(t)
	mockAgent.EXPECT().Instructions().Return("").Once()
	mockAgent.EXPECT().Messages().Return([]contractsai.Message{
		{Role: contractsai.RoleUser, Content: "What is the weather in London?"},
		{Role: contractsai.RoleAssistant, ToolCalls: []contractsai.ToolCall{{ID: "toolu_1", Name: "get_weather", RawArgs: `{"city":"London"}`}}},
		{Role: contractsai.RoleToolResult, ToolCallID: "toolu_1", Content: "Sunny, 25C"},
	}).Once()

	captured := make(chan capturedRequest, 1)
	server := newMessageServer(t, httpStatusOK, anthropicMessageResponse(t, []map[string]any{{"type": "text", "text": "done"}}, 5, 3), captured)
	defer server.Close()

	provider := &Provider{
		client: goanthropic.NewClient(option.WithoutEnvironmentDefaults(), option.WithBaseURL(server.URL), option.WithAPIKey("test-key")),
		config: contractsai.ProviderConfig{},
	}
	provider.config.Models.Text.Default = "claude-default"
	provider.config.Models.Text.MaxTokens = defaultMaxTokens

	_, err := provider.Prompt(context.Background(), contractsai.AgentPrompt{Agent: mockAgent, Input: "Thanks"})
	require.NoError(t, err)

	req := <-captured
	messages := req.body["messages"].([]any)
	require.Len(t, messages, 4)

	assistant := messages[1].(map[string]any)
	assert.Equal(t, "assistant", assistant["role"])
	assistantBlocks := assistant["content"].([]any)
	require.Len(t, assistantBlocks, 1)
	assert.Equal(t, "tool_use", assistantBlocks[0].(map[string]any)["type"])
	assert.Equal(t, "toolu_1", assistantBlocks[0].(map[string]any)["id"])

	toolResult := messages[2].(map[string]any)
	assert.Equal(t, "user", toolResult["role"])
	toolResultBlocks := toolResult["content"].([]any)
	require.Len(t, toolResultBlocks, 1)
	assert.Equal(t, "tool_result", toolResultBlocks[0].(map[string]any)["type"])
	assert.Equal(t, "toolu_1", toolResultBlocks[0].(map[string]any)["tool_use_id"])

	finalUser := messages[3].(map[string]any)
	assert.Equal(t, "user", finalUser["role"])
}

func TestProviderStreamPreservesToolArgsFromStartEvent(t *testing.T) {
	mockAgent := mocksai.NewAgent(t)
	mockAgent.EXPECT().Instructions().Return("").Once()
	mockAgent.EXPECT().Messages().Return(nil).Once()

	captured := make(chan capturedRequest, 1)
	server := newStreamingMessageServer(t, strings.Join([]string{
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{\"city\":\"London\"}}}\n",
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":\"\",\"container\":{},\"stop_details\":{}},\"usage\":{\"input_tokens\":4,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0,\"output_tokens\":2,\"output_tokens_details\":{\"thinking_tokens\":0},\"server_tool_use\":{\"web_fetch_requests\":0,\"web_search_requests\":0}}}\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n",
	}, "\n"), captured)
	defer server.Close()

	provider := &Provider{
		client: goanthropic.NewClient(option.WithoutEnvironmentDefaults(), option.WithBaseURL(server.URL), option.WithAPIKey("test-key")),
		config: contractsai.ProviderConfig{},
	}
	provider.config.Models.Text.Default = "claude-default"
	provider.config.Models.Text.MaxTokens = defaultMaxTokens

	stream, err := provider.Stream(context.Background(), contractsai.AgentPrompt{Agent: mockAgent, Input: "Use a tool"})
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
	assert.Empty(t, response.Text())
	require.Len(t, response.ToolCalls(), 1)
	assert.Equal(t, contractsai.ToolCall{ID: "toolu_1", Name: "get_weather", Args: map[string]any{"city": "London"}, RawArgs: `{"city":"London"}`}, response.ToolCalls()[0])
	require.Len(t, events, 1)
	assert.Equal(t, contractsai.StreamEventTypeDone, events[0].Type)
	assert.Equal(t, 6, events[0].Usage.Total())
}

const httpStatusOK = 200
