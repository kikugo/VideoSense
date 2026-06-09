from src.runtime_env import apply_secrets_to_env


def test_apply_secrets_sets_missing_keys():
    env: dict[str, str] = {}
    apply_secrets_to_env({'GEMINI_API_KEY': 'abc', 'VIDEOSENSE_QDRANT_URL': 'http://x'}, env)
    assert env == {'GEMINI_API_KEY': 'abc', 'VIDEOSENSE_QDRANT_URL': 'http://x'}


def test_apply_secrets_does_not_override_existing_env():
    env = {'GEMINI_API_KEY': 'from-real-env'}
    apply_secrets_to_env({'GEMINI_API_KEY': 'from-secrets'}, env)
    assert env['GEMINI_API_KEY'] == 'from-real-env'


def test_apply_secrets_skips_non_string_values():
    env: dict[str, str] = {}
    apply_secrets_to_env({'KEY': 'v', 'NUM': 5, 'SECTION': {'nested': 'x'}}, env)
    assert env == {'KEY': 'v'}
