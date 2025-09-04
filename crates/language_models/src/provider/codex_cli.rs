use crate::{AllLanguageModelSettings, CodexCliSettings};
use anyhow::{anyhow, Result};
use futures::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use futures::{
    future::BoxFuture,
    stream::{self, BoxStream},
    FutureExt, StreamExt,
};
use gpui::{AnyView, App, AppContext, AsyncApp, Context, EmptyView, Task, Window};
use language_model::{
    AuthenticateError, LanguageModel, LanguageModelCompletionError, LanguageModelCompletionEvent,
    LanguageModelId, LanguageModelName, LanguageModelProvider, LanguageModelProviderId,
    LanguageModelProviderName, LanguageModelProviderState, LanguageModelRequest,
    LanguageModelToolChoice, LanguageModelToolUse, LanguageModelToolUseId, Role, StopReason,
};
use serde::Deserialize;
use settings::Settings;
use std::path::PathBuf;
use std::sync::Arc;
use ui::IconName;
use util::{command::new_smol_command, paths};

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("codex-cli");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("Codex CLI");
const CODEX_CLI_SITE: &str = "https://github.com/microsoft/Codex-CLI";

pub struct CodexCliLanguageModelProvider {
    state: gpui::Entity<State>,
}

#[derive(Default)]
pub struct State {
    api_key: Option<String>,
}

impl State {
    fn is_authenticated(&self) -> bool {
        self.api_key.is_some()
    }

    fn authenticate(&self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        if self.is_authenticated() {
            return Task::ready(Ok(()));
        }

        cx.spawn(async move |this, cx| {
            let mut path = PathBuf::from(paths::home_dir());
            path.push(".codex");
            path.push("config.toml");

            let contents = match smol::fs::read_to_string(&path).await {
                Ok(c) => c,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                    return Err(AuthenticateError::CredentialsNotFound);
                }
                Err(err) => return Err(AuthenticateError::Other(err.into())),
            };

            #[derive(Deserialize)]
            struct CodexConfig {
                api_key: Option<String>,
            }
            let config: CodexConfig =
                toml::from_str(&contents).map_err(|e| AuthenticateError::Other(e.into()))?;
            let api_key = config
                .api_key
                .ok_or(AuthenticateError::CredentialsNotFound)?;

            this.update(cx, |this, cx| {
                this.api_key = Some(api_key);
                cx.notify();
            })?;

            Ok(())
        })
    }
}

impl CodexCliLanguageModelProvider {
    pub fn new(cx: &mut App) -> Self {
        let state = cx.new(|_| State { api_key: None });
        Self { state }
    }

    fn create_model(&self, cx: &App) -> Arc<dyn LanguageModel> {
        let settings = AllLanguageModelSettings::get_global(cx).codex_cli.clone();
        Arc::new(CodexCliLanguageModel {
            id: LanguageModelId::from("codex-cli".to_string()),
            name: LanguageModelName::from("Codex CLI".to_string()),
            model: "default".to_string(),
            state: self.state.clone(),
            settings,
        })
    }
}

impl LanguageModelProviderState for CodexCliLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<gpui::Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for CodexCliLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }
    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }
    fn icon(&self) -> IconName {
        IconName::Ai
    }

    fn default_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_model(cx))
    }

    fn default_fast_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_model(cx))
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        vec![self.create_model(cx)]
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| state.authenticate(cx))
    }

    fn configuration_view(
        &self,
        _target_agent: language_model::ConfigurationViewTargetAgent,
        _window: &mut Window,
        cx: &mut App,
    ) -> AnyView {
        cx.new(|_| EmptyView).into()
    }

    fn reset_credentials(&self, cx: &mut App) -> Task<Result<()>> {
        self.state.update(cx, |state, cx| {
            state.api_key = None;
            cx.notify();
        });
        Task::ready(Ok(()))
    }
}

struct CodexCliLanguageModel {
    id: LanguageModelId,
    name: LanguageModelName,
    model: String,
    state: gpui::Entity<State>,
    settings: CodexCliSettings,
}

impl LanguageModel for CodexCliLanguageModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }
    fn name(&self) -> LanguageModelName {
        self.name.clone()
    }
    fn provider_id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }
    fn provider_name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn telemetry_id(&self) -> String {
        format!("codex-cli/{}", self.model)
    }

    fn supports_images(&self) -> bool {
        false
    }
    fn supports_tools(&self) -> bool {
        true
    }
    fn supports_tool_choice(&self, choice: LanguageModelToolChoice) -> bool {
        matches!(
            choice,
            LanguageModelToolChoice::Auto | LanguageModelToolChoice::None
        )
    }
    fn max_token_count(&self) -> u64 {
        0
    }

    fn count_tokens(
        &self,
        _request: LanguageModelRequest,
        _cx: &App,
    ) -> BoxFuture<'static, Result<u64>> {
        async move { Ok(0) }.boxed()
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            BoxStream<'static, Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
            LanguageModelCompletionError,
        >,
    > {
        let state = self.state.clone();
        let model = self.model.clone();
        let settings = self.settings.clone();

        let api_key = match cx.read_entity(&state, |state, _| state.api_key.clone()) {
            Ok(Some(key)) => key,
            Ok(None) => {
                return futures::future::ready(Err(LanguageModelCompletionError::NoApiKey {
                    provider: PROVIDER_NAME,
                }))
                .boxed();
            }
            Err(_) => {
                return futures::future::ready(Err(LanguageModelCompletionError::Other(anyhow!(
                    "App state dropped",
                ))))
                .boxed();
            }
        };

        async move {
            let mut cmd = new_smol_command(&settings.binary_path);
            cmd.arg("exec").arg("--model").arg(&model);
            for (name, server) in &settings.mcp_servers {
                cmd.arg("--config").arg(format!(
                    "mcp_servers.{name}.command={}",
                    toml::Value::String(server.command.clone())
                ));
                if !server.args.is_empty() {
                    let args_val = toml::Value::Array(
                        server
                            .args
                            .iter()
                            .cloned()
                            .map(toml::Value::String)
                            .collect(),
                    );
                    cmd.arg("--config")
                        .arg(format!("mcp_servers.{name}.args={args_val}"));
                }
                if !server.env.is_empty() {
                    let env_val = toml::Value::Table(
                        server
                            .env
                            .iter()
                            .map(|(k, v)| (k.clone(), toml::Value::String(v.clone())))
                            .collect(),
                    );
                    cmd.arg("--config")
                        .arg(format!("mcp_servers.{name}.env={env_val}"));
                }
            }
            cmd.env("CODEX_API_KEY", api_key);
            cmd.stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .kill_on_drop(true);

            let mut child = cmd
                .spawn()
                .map_err(|e| LanguageModelCompletionError::Other(e.into()))?;

            let mut stdin = child.stdin.take().unwrap();
            let prompt = request
                .messages
                .iter()
                .map(|m| {
                    let role = match m.role {
                        Role::System => "system:",
                        Role::User => "user:",
                        Role::Assistant => "assistant:",
                    };
                    format!("{} {}\n", role, m.string_contents())
                })
                .collect::<String>();
            stdin
                .write_all(prompt.as_bytes())
                .await
                .map_err(|e| LanguageModelCompletionError::Other(e.into()))?;
            drop(stdin);

            let stdout = child.stdout.take().unwrap();
            let lines = BufReader::new(stdout).lines();

            let stream = stream::unfold((lines, false), |(mut lines, done)| async move {
                if done {
                    None
                } else {
                    match lines.next().await {
                        Some(Ok(line)) => {
                            let event = parse_line(&line);
                            Some((Ok(event), (lines, false)))
                        }
                        Some(Err(e)) => {
                            let err = LanguageModelCompletionError::ApiReadResponseError {
                                provider: PROVIDER_NAME,
                                error: e,
                            };
                            Some((Err(err), (lines, true)))
                        }
                        None => Some((
                            Ok(LanguageModelCompletionEvent::Stop(StopReason::EndTurn)),
                            (lines, true),
                        )),
                    }
                }
            });

            Ok(Box::pin(stream) as BoxStream<_>)
        }
        .boxed()
    }
}

fn parse_line(line: &str) -> LanguageModelCompletionEvent {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
        if let Some(r#type) = value.get("type").and_then(|v| v.as_str()) {
            if r#type == "tool" || r#type == "tool_use" {
                if let (Some(id), Some(name), Some(input)) = (
                    value.get("id").and_then(|v| v.as_str()),
                    value
                        .get("name")
                        .or_else(|| value.get("tool_name"))
                        .and_then(|v| v.as_str()),
                    value.get("input"),
                ) {
                    let raw_input = value
                        .get("raw_input")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let is_input_complete = value
                        .get("is_input_complete")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true);
                    return LanguageModelCompletionEvent::ToolUse(LanguageModelToolUse {
                        id: LanguageModelToolUseId::from(id.to_string()),
                        name: name.into(),
                        raw_input,
                        input: input.clone(),
                        is_input_complete,
                    });
                }
            }
        }
        if let Some(text) = value.get("content").and_then(|v| v.as_str()) {
            return LanguageModelCompletionEvent::Text(text.to_string());
        }
        if let Some(text) = value.get("text").and_then(|v| v.as_str()) {
            return LanguageModelCompletionEvent::Text(text.to_string());
        }
    }
    LanguageModelCompletionEvent::Text(line.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::TestAppContext;
    use language_model::LanguageModelRegistry;

    #[gpui::test]
    fn registers_with_language_model_registry(cx: &mut TestAppContext) {
        // Initialize a test registry and register the Codex CLI provider
        cx.update(|app| {
            LanguageModelRegistry::test(app);
            let registry = LanguageModelRegistry::global(app);
            registry.update(app, |registry, cx| {
                registry.register_provider(CodexCliLanguageModelProvider::new(cx), cx);
            });
        });

        // Ensure the provider is now part of the registry
        let is_registered = cx.update(|app| {
            LanguageModelRegistry::read_global(app)
                .providers()
                .iter()
                .any(|p| p.id() == PROVIDER_ID)
        });
        assert!(is_registered);
    }
}
