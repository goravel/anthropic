package facades

import (
	"fmt"
	"testing"

	contractsai "github.com/goravel/framework/contracts/ai"
	mocksai "github.com/goravel/framework/mocks/ai"
	mocksfoundation "github.com/goravel/framework/mocks/foundation"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	root "github.com/goravel/anthropic"
)

func TestAnthropic(t *testing.T) {
	t.Run("returns error when service provider is not registered", func(t *testing.T) {
		previousApp := root.App
		root.App = nil
		defer func() { root.App = previousApp }()

		provider, err := Anthropic("anthropic")

		require.Error(t, err)
		assert.Nil(t, provider)
		assert.Equal(t, "please register anthropic service provider", err.Error())
	})

	t.Run("returns provider from binding", func(t *testing.T) {
		previousApp := root.App
		defer func() { root.App = previousApp }()

		app := mocksfoundation.NewApplication(t)
		provider := mocksai.NewProvider(t)
		app.EXPECT().MakeWith(root.Binding, map[string]any{"provider": "anthropic"}).Return(contractsai.Provider(provider), nil).Once()
		root.App = app

		resolvedProvider, err := Anthropic("anthropic")

		require.NoError(t, err)
		assert.Equal(t, contractsai.Provider(provider), resolvedProvider)
	})

	t.Run("returns error for unexpected binding type", func(t *testing.T) {
		previousApp := root.App
		defer func() { root.App = previousApp }()

		app := mocksfoundation.NewApplication(t)
		app.EXPECT().MakeWith(root.Binding, map[string]any{"provider": "anthropic"}).Return("wrong-type", nil).Once()
		root.App = app

		provider, err := Anthropic("anthropic")

		require.Error(t, err)
		assert.Nil(t, provider)
		assert.Equal(t, fmt.Sprintf("anthropic binding returned %T, expected ai.Provider", "wrong-type"), err.Error())
	})
}
