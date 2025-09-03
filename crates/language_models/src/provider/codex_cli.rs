use crate::ui::InstructionListItem;
use anyhow::{Context as _, Result, anyhow};
use futures::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use futures::{
    FutureExt, StreamExt,
    future::BoxFuture,
    stream::{self, BoxStream},
};
use gpui::{AnyView, App, AsyncApp, Context, Task, Window};
use language_model::{
    AuthenticateError, LanguageModel, LanguageModelCompletionError, LanguageModelCompletionEvent,
    LanguageModelId, LanguageModelName, LanguageModelProvider, LanguageModelProviderId,
    LanguageModelProviderName, LanguageModelProviderState, LanguageModelRequest,
    LanguageModelToolChoice, Role, StopReason,
};
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;
use ui::{List, prelude::*};
use util::{command::new_smol_command, paths};

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("codex-cli");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("Codex CLI");
const CODEX_CLI_SITE: &str = "https://github.com/microsoft/Codex-CLI";

#[derive(Default)]
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

        cx.spawn(|this, cx| async move {
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

    fn create_model(&self) -> Arc<dyn LanguageModel> {
        Arc::new(CodexCliLanguageModel {
            id: LanguageModelId::from("codex-cli".to_string()),
            name: LanguageModelName::from("Codex CLI".to_string()),
            model: "default".to_string(),
            state: self.state.clone(),
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

    fn default_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_model())
    }

    fn default_fast_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_model())
    }

    fn provided_models(&self, _cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        vec![self.create_model()]
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
        cx.new(|cx| {
            v_flex()
                .gap_2()
                .child(
                    v_flex().gap_1().child(Label::new(
                        "Codex CLI uses `~/.codex/config.toml` for authentication.",
                    ))
                    .child(
                        List::new()
                            .child(InstructionListItem::text_only(
                                "Install the `codex` binary and ensure it is available on your PATH.",
                            ))
                            .child(InstructionListItem::text_only(
                                "Authenticate by running `codex auth login` or by adding `api_key` to `~/.codex/config.toml`.",
                            )),
                    ),
                )
                .child(
                    Button::new("codex-cli-site", "Codex CLI")
                        .style(ButtonStyle::Subtle)
                        .icon(IconName::ArrowUpRight)
                        .icon_size(IconSize::Small)
                        .icon_color(Color::Muted)
                        .on_click(move |_, _window, cx| {
                            cx.open_url(CODEX_CLI_SITE);
                        }),
                )
        })
        .into()
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
        false
    }
    fn supports_tool_choice(&self, _choice: LanguageModelToolChoice) -> bool {
        false
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
        async move {
            let api_key = cx
                .read_entity(&state, |state, _| state.api_key.clone())
                .ok_or_else(|| LanguageModelCompletionError::Other(anyhow!("App state dropped")))?;
            let api_key = api_key.ok_or(LanguageModelCompletionError::NoApiKey {
                provider: PROVIDER_NAME,
            })?;

            let mut cmd = new_smol_command("codex");
            cmd.arg("exec").arg("--model").arg(&model);
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
                        Role::Tool => "tool:",
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
        if let Some(text) = value.get("content").and_then(|v| v.as_str()) {
            return LanguageModelCompletionEvent::Text(text.to_string());
        }
        if let Some(text) = value.get("text").and_then(|v| v.as_str()) {
            return LanguageModelCompletionEvent::Text(text.to_string());
        }
    }
    LanguageModelCompletionEvent::Text(line.to_string())
}
