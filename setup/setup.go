package main

import (
	"os"

	"github.com/goravel/framework/packages"
	"github.com/goravel/framework/packages/match"
	"github.com/goravel/framework/packages/modify"
	"github.com/goravel/framework/support/path"
)

func main() {
	setup := packages.Setup(os.Args)
	aiConfigPath := path.Config("ai.go")
	moduleImport := setup.Paths().Module().Import()
	serviceProvider := "&anthropic.ServiceProvider{}"
	aiProviderContract := "github.com/goravel/framework/contracts/ai"
	anthropicFacadesImport := moduleImport + "/facades"
	provider := `map[string]any{
		"key": config.Env("ANTHROPIC_API_KEY", ""),
		"models": map[string]any{
			"text": map[string]any{
				"default": "",
				"max_tokens": 4096,
			},
		},
		"url": config.Env("ANTHROPIC_API_URL", ""),
		"via": func() (ai.Provider, error) {
			return anthropicfacades.Anthropic("anthropic")
		},
	}`
	aiProvidersConfig := match.Config("ai.providers")

	setup.Install(
		modify.RegisterProvider(moduleImport, serviceProvider),

		modify.GoFile(aiConfigPath).Find(match.Imports()).Modify(
			modify.AddImport(aiProviderContract),
			modify.AddImport(anthropicFacadesImport, "anthropicfacades"),
		).Find(aiProvidersConfig).Modify(modify.AddConfig("anthropic", provider)),
	).Uninstall(
		modify.WhenFileExists(aiConfigPath, modify.GoFile(aiConfigPath).
			Find(aiProvidersConfig).Modify(modify.RemoveConfig("anthropic")).
			Find(match.Imports()).Modify(
			modify.RemoveImport(aiProviderContract),
			modify.RemoveImport(anthropicFacadesImport, "anthropicfacades"),
		)),

		modify.UnregisterProvider(moduleImport, serviceProvider),
	).Execute()
}
