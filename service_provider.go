package anthropic

import (
	"github.com/goravel/framework/contracts/ai"
	"github.com/goravel/framework/contracts/binding"
	"github.com/goravel/framework/contracts/foundation"
	"github.com/goravel/framework/errors"
)

const (
	Binding = "goravel.anthropic"
	Name    = "Anthropic"
)

var App foundation.Application

type ServiceProvider struct{}

func (r *ServiceProvider) Relationship() binding.Relationship {
	return binding.Relationship{
		Bindings: []string{
			Binding,
		},
		Dependencies: []string{
			binding.Config,
		},
		ProvideFor: []string{
			binding.AI,
		},
	}
}

func (r *ServiceProvider) Register(app foundation.Application) {
	App = app

	app.BindWith(Binding, func(app foundation.Application, parameters map[string]any) (any, error) {
		config := app.MakeConfig()
		if config == nil {
			return nil, errors.ConfigFacadeNotSet.SetModule(Name)
		}

		provider, err := NewAnthropic(config, parameters["provider"].(string))
		if err != nil {
			return nil, err
		}

		return ai.Provider(provider), nil
	})
}

func (r *ServiceProvider) Boot(app foundation.Application) {}
