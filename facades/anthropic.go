package facades

import (
	"fmt"

	contractsai "github.com/goravel/framework/contracts/ai"

	"github.com/goravel/anthropic"
)

func Anthropic(provider string) (contractsai.Provider, error) {
	if anthropic.App == nil {
		return nil, fmt.Errorf("please register anthropic service provider")
	}

	instance, err := anthropic.App.MakeWith(anthropic.Binding, map[string]any{
		"provider": provider,
	})
	if err != nil {
		return nil, err
	}

	return instance.(contractsai.Provider), nil
}
