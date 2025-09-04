use std::env;

use gpui::TestAppContext;
use language_model::{AuthenticateError, LanguageModelProvider};
use language_models::provider::codex_cli::CodexCliLanguageModelProvider;
use tempfile::tempdir;

#[gpui::test]
async fn authentication_error_when_no_credentials(cx: &mut TestAppContext) {
    let home = tempdir().unwrap();
    unsafe {
        env::set_var("HOME", home.path());
    }

    cx.update(|app| {
        let store = settings::SettingsStore::test(app);
        app.set_global(store);
        language_models::init_settings(app);
    });
    let task = cx.update(|app| {
        let provider = CodexCliLanguageModelProvider::new(app);
        provider.authenticate(app)
    });

    match task.await {
        Err(AuthenticateError::CredentialsNotFound) => {}
        other => panic!("expected CredentialsNotFound, got {:?}", other),
    }
}
