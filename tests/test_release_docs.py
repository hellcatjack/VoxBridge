from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = (
    ROOT / "README.md",
    ROOT / "CHANGELOG.md",
    ROOT / "docs" / "API.md",
    ROOT / "docs" / "DEPLOYMENT.md",
    ROOT / "docs" / "SECURITY_SCAN.md",
)
USER_DOCS = PUBLIC_DOCS[:-1]


def test_release_version_is_0_2_0():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["version"] == "0.2.0"


def test_public_release_documents_exist_and_are_linked():
    missing = [str(path.relative_to(ROOT)) for path in PUBLIC_DOCS if not path.is_file()]
    assert missing == []
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for relative_path in (
        "CHANGELOG.md",
        "docs/API.md",
        "docs/DEPLOYMENT.md",
        "docs/SECURITY_SCAN.md",
    ):
        assert relative_path in readme


def test_public_user_documents_are_portable_and_sanitized():
    private_ipv4 = re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|"
        r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
    )
    absolute_workspace = re.compile(r"/(?:data|home)/[A-Za-z0-9._/-]+")
    for path in USER_DOCS:
        text = path.read_text(encoding="utf-8")
        assert private_ipv4.search(text) is None, path
        assert absolute_workspace.search(text) is None, path


def test_runtime_and_internal_artifacts_remain_ignored():
    ignored = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
    assert {".venv/", "logs/", "*.log", "docs/superpowers/"} <= ignored


def test_readme_declares_the_supported_runtime_contract():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "8024" in readme
    assert ".venv/bin/python" in readme
    assert "Qwen/Qwen3-ASR-0.6B" in readme
    assert "sentence_id" in readme
    assert "revision" in readme


def test_install_docs_use_uv_with_the_project_venv():
    install_docs = (ROOT / "README.md", ROOT / "docs" / "DEPLOYMENT.md")
    for path in install_docs:
        text = path.read_text(encoding="utf-8")
        assert ".venv/bin/python -m pip" not in text, path
        assert "uv pip install --python .venv/bin/python -e ." in text, path


def test_public_docs_describe_esv_zh_en_translation_policy():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    api = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert "ESV" in readme
    assert "ESV" in api
    assert "ESV" in changelog
    assert "不补全" in readme


def test_public_docs_describe_optional_kokoro_tts_contract():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    api = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    deployment = (ROOT / "docs" / "DEPLOYMENT.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert "Kokoro-82M" in readme
    assert "CPU-only" in readme
    assert "FIFO" in readme
    assert "/listen" in readme
    assert "仅接收连接后" in readme
    assert "多个设备" in readme
    assert "主字幕页不播放" in readme
    assert "0.8x" in readme
    assert "1.2x" in readme
    assert "0.75x" not in readme
    assert "2.0x" not in readme
    assert "单条预取" in readme
    assert "uv pip install --python .venv/bin/python -e '.[tts]'" in readme

    assert "GET /listen" in api
    assert "WS /ws/tts" in api
    assert "POST /api/tts/broadcast/jobs/{job_id}/audio" in api
    assert '"type": "tts_received"' in api
    assert "future-only" in api
    assert "deprecated" in api
    assert "POST /api/tts/jobs/{job_id}/audio" in api
    assert "DELETE /api/tts/jobs/{job_id}" in api
    assert "DELETE /api/tts/clients/{client_id}/jobs" in api
    assert '"type": "tts_job"' in api
    assert '"is_stable": true' in api
    assert "per-device" in api.lower()
    assert "localStorage" in api
    assert "single-item lookahead" in api

    assert "--enable-tts" in deployment
    assert "--tts-en-model-path" in deployment
    assert "--tts-zh-model-path" in deployment
    assert "--tts-listener-queue-size" in deployment
    assert "--tts-speed" in deployment
    assert "listener-side" in deployment.lower()
    assert "bounded" in deployment.lower()
    assert "CPUExecutionProvider" in deployment
    assert "/listen" in deployment
    assert "Kokoro" in changelog
    assert "multi-listener" in changelog


def test_docs_publish_tts_revision_stability_contract():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    api = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    deployment = (ROOT / "docs" / "DEPLOYMENT.md").read_text(encoding="utf-8")

    assert "--tts-revision-stable-sec" in api
    assert "--tts-revision-stable-sec 3.0" in deployment
    assert "source revision" in api.lower()
    assert "字幕" in readme and "朗读" in readme
    assert "3.0" in readme
