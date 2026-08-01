from __future__ import annotations

import json

import app as app_module


def test_sample_manifest_is_present_and_playable():
    """The bundled clip must actually ship, or the empty state offers a dead button."""
    manifest = app_module.load_sample_manifest()

    assert manifest is not None, 'examples/sample manifest missing or its video file is absent'
    assert (app_module.SAMPLE_DIR / manifest['file']).exists()
    assert manifest['try_asking'], 'sample needs suggested queries to be useful as a demo'


def test_sample_manifest_credits_its_source():
    """This file is redistributed publicly, so provenance has to travel with it."""
    manifest = app_module.load_sample_manifest()

    assert manifest['credit']
    assert manifest['license']
    assert manifest['source_url'].startswith('https://')


def test_sample_video_stays_small_enough_to_commit():
    manifest = app_module.load_sample_manifest()
    size_mb = (app_module.SAMPLE_DIR / manifest['file']).stat().st_size / (1024 * 1024)

    assert size_mb < 10, f'sample video grew to {size_mb:.1f} MB; keep the repo clone cheap'


def test_load_sample_manifest_returns_none_when_video_missing(tmp_path, monkeypatch):
    """A manifest without its video must not advertise a button that cannot work."""
    manifest_dir = tmp_path / 'sample'
    manifest_dir.mkdir()
    (manifest_dir / 'manifest.json').write_text(json.dumps({'file': 'absent.mp4'}), encoding='utf-8')

    monkeypatch.setattr(app_module, 'SAMPLE_DIR', manifest_dir)
    monkeypatch.setattr(app_module, 'SAMPLE_MANIFEST', manifest_dir / 'manifest.json')

    assert app_module.load_sample_manifest() is None


def test_load_sample_manifest_returns_none_when_manifest_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, 'SAMPLE_DIR', tmp_path)
    monkeypatch.setattr(app_module, 'SAMPLE_MANIFEST', tmp_path / 'does-not-exist.json')

    assert app_module.load_sample_manifest() is None
