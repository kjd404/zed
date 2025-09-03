use std::{env, fs};

use futures::StreamExt;
use gpui::TestAppContext;
use language_model::{
    LanguageModelCompletionEvent, LanguageModelCompletionError, LanguageModelProviderId,
    LanguageModelRegistry, LanguageModelRequest, LanguageModelRequestMessage, MessageContent,
    Role,
};
use language_models::provider::codex_cli::CodexCliLanguageModelProvider;
use tempfile::tempdir;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

#[gpui::test]
async fn handles_request_response(cx: &mut TestAppContext) {
    // Configure HOME with Codex credentials
    let home = tempdir().unwrap();
    env::set_var("HOME", home.path());
    let config_dir = home.path().join(".codex");
    fs::create_dir_all(&config_dir).unwrap();
    fs::write(config_dir.join("config.toml"), "api_key = \"test\"\n").unwrap();

    // Mock `codex` binary
    let bin_dir = tempdir().unwrap();
    let script_path = bin_dir.path().join("codex");
    fs::write(
        &script_path,
        "#!/bin/sh\ncat >/dev/null\necho '{\"content\":\"hello\"}'\n",
    )
    .unwrap();
    #[cfg(unix)]
    fs::set_permissions(&script_path, fs::Permissions::from_mode(0o755)).unwrap();
    env::set_var("PATH", bin_dir.path());

    // Register provider and authenticate
    let provider_id = LanguageModelProviderId::new("codex-cli");
    let (auth_task, model) = cx.update(|app| {
        LanguageModelRegistry::test(app);
        let registry = LanguageModelRegistry::global(app);
        registry.update(app, |registry, cx| {
            registry.register_provider(CodexCliLanguageModelProvider::new(cx), cx);
        });
        let provider = LanguageModelRegistry::read_global(app)
            .providers()
            .into_iter()
            .find(|p| p.id() == provider_id)
            .unwrap();
        let auth_task = provider.authenticate(app);
        let model = provider.default_model(app).unwrap();
        (auth_task, model)
    });
    auth_task.await.unwrap();

    let request = LanguageModelRequest {
        messages: vec![LanguageModelRequestMessage {
            role: Role::User,
            content: vec![MessageContent::from("hello")],
            cache: false,
        }],
        ..Default::default()
    };

    let async_cx = cx.to_async();
    let stream = model.stream_completion(request, &async_cx).await.unwrap();
    let events: Vec<_> = stream.collect().await;
    assert_eq!(
        events[0],
        Ok(LanguageModelCompletionEvent::Text("hello".to_string()))
    );
}

#[gpui::test]
async fn errors_when_binary_missing(cx: &mut TestAppContext) {
    // Configure HOME with Codex credentials
    let home = tempdir().unwrap();
    env::set_var("HOME", home.path());
    let config_dir = home.path().join(".codex");
    fs::create_dir_all(&config_dir).unwrap();
    fs::write(config_dir.join("config.toml"), "api_key = \"test\"\n").unwrap();

    // PATH without `codex`
    env::set_var("PATH", "");

    // Register provider and authenticate
    let provider_id = LanguageModelProviderId::new("codex-cli");
    let (auth_task, model) = cx.update(|app| {
        LanguageModelRegistry::test(app);
        let registry = LanguageModelRegistry::global(app);
        registry.update(app, |registry, cx| {
            registry.register_provider(CodexCliLanguageModelProvider::new(cx), cx);
        });
        let provider = LanguageModelRegistry::read_global(app)
            .providers()
            .into_iter()
            .find(|p| p.id() == provider_id)
            .unwrap();
        let auth_task = provider.authenticate(app);
        let model = provider.default_model(app).unwrap();
        (auth_task, model)
    });
    auth_task.await.unwrap();

    let request = LanguageModelRequest {
        messages: vec![LanguageModelRequestMessage {
            role: Role::User,
            content: vec![MessageContent::from("hello")],
            cache: false,
        }],
        ..Default::default()
    };

    let async_cx = cx.to_async();
    let result = model.stream_completion(request, &async_cx).await;
    assert!(matches!(result, Err(LanguageModelCompletionError::Other(_))));
}

