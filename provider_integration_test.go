package anthropic

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	contractsai "github.com/goravel/framework/contracts/ai"
	mocksai "github.com/goravel/framework/mocks/ai"
	mocksconfig "github.com/goravel/framework/mocks/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProviderPromptIntegration(t *testing.T) {
	apiKey := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY"))
	if apiKey == "" {
		t.Skip("ANTHROPIC_API_KEY is not set")
	}

	mockConfig := mocksconfig.NewConfig(t)
	mockConfig.EXPECT().UnmarshalKey("ai.providers.anthropic", new(contractsai.ProviderConfig)).RunAndReturn(func(_ string, rawVal any) error {
		cfg := rawVal.(*contractsai.ProviderConfig)
		cfg.Key = apiKey
		cfg.Models.Text.MaxTokens = defaultMaxTokens
		return nil
	}).Once()

	mockAgent := mocksai.NewAgent(t)
	mockAgent.EXPECT().Instructions().Return("").Once()
	mockAgent.EXPECT().Messages().Return(nil).Once()

	provider, err := NewAnthropic(mockConfig, "anthropic")
	require.NoError(t, err)
	require.NotNil(t, provider)

	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()

	response, err := provider.Prompt(ctx, contractsai.AgentPrompt{
		Agent: mockAgent,
		Input: "Reply with OK only. Do not include punctuation or extra text.",
	})
	require.NoError(t, err)
	require.NotNil(t, response)
	require.NotNil(t, response.Usage())

	assert.Equal(t, "OK", strings.TrimSpace(response.Text()))
	assert.Positive(t, response.Usage().Total())
	assert.Empty(t, response.ToolCalls())
}
