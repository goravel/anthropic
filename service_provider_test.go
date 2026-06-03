package anthropic

import (
	"testing"

	contractsai "github.com/goravel/framework/contracts/ai"
	contractsfoundation "github.com/goravel/framework/contracts/foundation"
	mocksconfig "github.com/goravel/framework/mocks/config"
	mocksfoundation "github.com/goravel/framework/mocks/foundation"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestServiceProviderRegisterReturnsErrorForMissingProviderParameter(t *testing.T) {
	app := mockServiceProviderApp(t, func(_ string, _ *contractsai.ProviderConfig) error {
		return nil
	})

	callback := registerCallback(t, app)
	instance, err := callback(app, map[string]any{})

	require.Error(t, err)
	assert.Nil(t, instance)
	assert.Equal(t, "missing anthropic provider parameter", err.Error())
}

func TestServiceProviderRegisterBuildsProviderFromParameter(t *testing.T) {
	app := mockServiceProviderApp(t, func(key string, cfg *contractsai.ProviderConfig) error {
		assert.Equal(t, "ai.providers.anthropic", key)
		cfg.Key = "test-key"
		return nil
	})

	callback := registerCallback(t, app)
	instance, err := callback(app, map[string]any{"provider": "anthropic"})

	require.NoError(t, err)
	_, ok := instance.(contractsai.Provider)
	assert.True(t, ok)
}

func mockServiceProviderApp(t *testing.T, unmarshal func(string, *contractsai.ProviderConfig) error) *mocksfoundation.Application {
	t.Helper()

	app := mocksfoundation.NewApplication(t)
	config := mocksconfig.NewConfig(t)
	config.EXPECT().UnmarshalKey("ai.providers.anthropic", new(contractsai.ProviderConfig)).RunAndReturn(func(key string, rawVal any) error {
		return unmarshal(key, rawVal.(*contractsai.ProviderConfig))
	}).Maybe()
	app.EXPECT().MakeConfig().Return(config).Maybe()

	return app
}

func registerCallback(t *testing.T, app *mocksfoundation.Application) func(contractsfoundation.Application, map[string]any) (any, error) {
	t.Helper()

	provider := &ServiceProvider{}
	var callback func(contractsfoundation.Application, map[string]any) (any, error)
	app.EXPECT().BindWith(Binding, mock.Anything).Run(func(_ any, bindingCallback func(contractsfoundation.Application, map[string]any) (any, error)) {
		callback = bindingCallback
	})

	provider.Register(app)
	require.NotNil(t, callback)

	return callback
}
