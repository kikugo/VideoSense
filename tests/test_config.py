from src.config import AppConfig


def test_app_config_reads_env_values(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'abc123')
    monkeypatch.setenv('VIDEOSENSE_FRAME_INTERVAL_SEC', '1.5')
    monkeypatch.setenv('VIDEOSENSE_TOP_K', '5')
    monkeypatch.setenv('VIDEOSENSE_EMBED_CONCURRENCY', '7')
    monkeypatch.setenv('VIDEOSENSE_ENABLE_PERSISTENCE', 'true')
    monkeypatch.setenv('VIDEOSENSE_MIN_SIMILARITY', '0.55')
    monkeypatch.setenv('VIDEOSENSE_FRAME_STRATEGY', 'interval')
    monkeypatch.setenv('VIDEOSENSE_SCENE_THRESHOLD', '30.0')
    monkeypatch.setenv('VIDEOSENSE_MAX_VISUAL_FRAMES', '20')

    config = AppConfig.from_env()

    assert config.gemini_api_key == 'abc123'
    assert config.frame_interval_sec == 1.5
    assert config.top_k == 5
    assert config.embed_concurrency == 7
    assert config.enable_persistence is True
    assert config.min_similarity == 0.55
    assert config.frame_strategy == 'interval'
    assert config.scene_threshold == 30.0
    assert config.max_visual_frames == 20


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
    assert config.min_similarity == 0.3
    assert config.frame_strategy == 'scene'
    assert config.scene_threshold == 27.0
    assert config.max_visual_frames == 300
