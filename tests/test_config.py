from src.config import AppConfig


def test_app_config_reads_env_values(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'abc123')
    monkeypatch.setenv('VIDEOSENSE_FRAME_INTERVAL_SEC', '1.5')
    monkeypatch.setenv('VIDEOSENSE_TOP_K', '5')
    monkeypatch.setenv('VIDEOSENSE_EMBED_CONCURRENCY', '7')
    monkeypatch.setenv('VIDEOSENSE_ENABLE_PERSISTENCE', 'true')

    config = AppConfig.from_env()

    assert config.gemini_api_key == 'abc123'
    assert config.frame_interval_sec == 1.5
    assert config.top_k == 5
    assert config.embed_concurrency == 7
    assert config.enable_persistence is True


def test_app_config_defaults_without_env(monkeypatch):
    monkeypatch.delenv('VIDEOSENSE_FRAME_INTERVAL_SEC', raising=False)
    monkeypatch.delenv('VIDEOSENSE_TOP_K', raising=False)
    monkeypatch.delenv('VIDEOSENSE_EMBED_CONCURRENCY', raising=False)
    monkeypatch.delenv('VIDEOSENSE_ENABLE_PERSISTENCE', raising=False)

    config = AppConfig.from_env()

    assert config.frame_interval_sec == 2.0
    assert config.top_k == 3
    assert config.embed_concurrency == 4
    assert config.enable_persistence is False
