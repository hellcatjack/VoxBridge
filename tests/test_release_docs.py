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
    assert "Qwen/Qwen3-ASR-1.7B" not in readme
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
    assert "bounded pre-listener backlog" in readme
    assert "只保留最新一条稳定译文" in readme
    assert "多个设备" in readme
    assert "主字幕页不播放" in readme
    assert "HLS" in readme
    assert "hls.js" in readme
    assert "锁屏" in readme
    assert "一个 Kokoro worker" in readme
    assert "https://ushome.amycat.com:18024/listen" in readme
    assert "免登录" in readme
    assert "Pittsburgh Christian Church South" in readme
    assert "1.2x" in readme
    assert "1.5x" in readme
    assert "--disable-tts-global-auto-speed" in readme
    assert "0.8x" not in readme
    assert "0.75x" not in readme
    assert "媒体播放率 2.0x" not in readme
    assert "300ms" in readme
    assert "pending_audio_ms" in readme
    assert "uv pip install --python .venv/bin/python -e '.[tts]'" in readme

    assert "GET /listen" in api
    assert "GET /listen/assets/hls.min.js" in api
    assert "hls.js" in api
    assert "GET /api/tts/live/status" in api
    assert '"synthesis_active"' in api
    assert '"preparation_queue_depth"' in api
    assert '"prepared_audio_count"' in api
    assert '"pending_audio_ms"' in api
    assert '"speech_epoch_id"' in api
    assert '"global_speed_multiplier"' in api
    assert '"tts_effective_speed"' in api
    assert "maxLiveSyncPlaybackRate: 1" in api
    assert "GET /api/tts/live/{listener_id}/index.m3u8" in api
    assert "GET /api/tts/live/{listener_id}/captions" in api
    assert '"live_edge_at_ms"' in api
    assert '"cues"' in api
    assert "256" in api
    assert "getStartDate() + currentTime" in api
    assert "PCM media timeline" in api
    assert "1024-sample" in api
    assert "edge silence" in api
    assert "does not gate HLS audio" in api
    assert "GET /api/tts/live/{listener_id}/segments/{segment_name}" in api
    assert "DELETE /api/tts/live/{listener_id}" in api
    assert "persistent media element" in api
    assert "public bearer capability" in api
    assert "HTTP 429" in api
    assert "128" in api
    assert "GET /listen/qr.svg" in api
    assert "Pittsburgh Christian Church South" in api
    assert "WS /ws/tts" in api
    assert "POST /api/tts/broadcast/jobs/{job_id}/audio" in api
    assert '"type": "tts_received"' in api
    assert "bounded pre-listener pool" in api
    assert "latest stable translation" in api
    assert "exact translation revision" in api
    assert "deprecated" in api
    assert "POST /api/tts/jobs/{job_id}/audio" in api
    assert "DELETE /api/tts/jobs/{job_id}" in api
    assert "DELETE /api/tts/clients/{client_id}/jobs" in api
    assert '"type": "tts_job"' in api
    assert '"is_stable": true' in api
    assert "read-only shared status" in api
    assert "per-device `Auto`" not in api
    assert "localStorage" not in api
    assert "discardable_gap_before_ms" in api
    assert "resume_at_ms" in api

    assert "--enable-tts" in deployment
    assert "--tts-en-model-path" in deployment
    assert "--tts-zh-model-path" in deployment
    assert "--tts-listener-queue-size" in deployment
    assert "--tts-speed" in deployment
    assert "--disable-tts-global-auto-speed" in deployment
    assert "server-paced shared stream" in deployment
    assert "bounded" in deployment.lower()
    assert "300ms" in deployment
    assert "one FFmpeg" in deployment
    assert "90-second lease" in deployment
    assert "6-<15s = 1.2x" in deployment
    assert "15-<20s = 1.4x" in deployment
    assert ">=20s = 1.5x" in deployment
    assert "live-edge" in deployment
    assert "caption cue" in deployment.lower()
    assert "getStartDate()" in deployment
    assert "PCM media timeline" in deployment
    assert "256" in deployment
    assert "--tts-hls-max-listeners 128" in deployment
    assert "bounded pre-listener pool" in deployment
    assert "retains only the latest stable" in deployment
    assert "public bearer capability" in deployment
    assert "two-second" in deployment.lower()
    assert "one-shot" in deployment.lower()
    assert "不循环重试" in readme
    assert "main subtitle page remains authenticated" in deployment
    assert "CPUExecutionProvider" in deployment
    assert "/listen" in deployment
    assert "hls.js" in deployment
    assert "Kokoro" in changelog
    assert "hls.js" in changelog
    assert "multi-listener" in changelog
    assert "PCCS" in changelog
    assert "static QR" in changelog


def test_docs_publish_tts_revision_stability_contract():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    api = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    deployment = (ROOT / "docs" / "DEPLOYMENT.md").read_text(encoding="utf-8")

    assert "--tts-revision-stable-sec" in api
    assert "--tts-revision-stable-sec 3.0" in deployment
    assert "source revision" in api.lower()
    assert "字幕" in readme and "朗读" in readme
    assert "3.0" in readme


def test_user_service_templates_bound_log_rotation():
    logrotate = (
        ROOT / "deploy" / "logrotate" / "voxbridge.conf"
    ).read_text(encoding="utf-8")
    service = (
        ROOT / "deploy" / "systemd" / "voxbridge-logrotate.service"
    ).read_text(encoding="utf-8")
    timer = (
        ROOT / "deploy" / "systemd" / "voxbridge-logrotate.timer"
    ).read_text(encoding="utf-8")

    assert "/data/Qwen3-ASR/logs/voxbridge_8024.log" in logrotate
    assert "/data/Qwen3-ASR/logs/voxbridge_subtitle_trace.jsonl" in logrotate
    assert "size 512M" in logrotate
    assert "rotate 21" in logrotate
    assert "compress" in logrotate
    assert "copytruncate" in logrotate
    assert "%h/.local/state/voxbridge/logrotate.status" in service
    assert "%h/.config/voxbridge/logrotate.conf" in service
    assert "OnUnitActiveSec=1h" in timer
    assert "Persistent=true" in timer


def test_user_service_couples_translation_lifecycle_and_readiness():
    translation = (
        ROOT / "deploy" / "systemd" / "voxbridge-translation.service"
    ).read_text(encoding="utf-8")
    dependency = (
        ROOT / "deploy" / "systemd" / "voxbridge-8024-translation.conf"
    ).read_text(encoding="utf-8")
    readiness = (
        ROOT / "deploy" / "systemd" / "voxbridge-wait-translation.sh"
    ).read_text(encoding="utf-8")
    deployment = (ROOT / "docs" / "DEPLOYMENT.md").read_text(encoding="utf-8")

    assert "Type=forking" in translation
    assert "PIDFile=/app/llama.cpp/build/bin/llama-server-normal.pid" in translation
    assert "ExecStart=/app/llama.cpp/llama-server-normal.sh start" in translation
    assert "ExecStop=/app/llama.cpp/llama-server-normal.sh stop" in translation
    assert "PartOf=voxbridge-8024.service" in translation
    assert "Before=voxbridge-8024.service" in translation
    assert "ExecStartPost=%h/.local/libexec/voxbridge-wait-translation" in translation
    assert "Requires=voxbridge-translation.service" in dependency
    assert "After=voxbridge-translation.service" in dependency
    assert "ExecStartPre" not in dependency
    assert "/v1/models" in readiness
    assert "tencent/HY-MT1.5-1.8B-GGUF:Q8_0" in readiness
    assert "systemctl --user start voxbridge-8024.service" in deployment
    assert "systemctl --user stop voxbridge-8024.service" in deployment


def test_docs_publish_runtime_budget_and_tts_finality_contract():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    api = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    deployment = (ROOT / "docs" / "DEPLOYMENT.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert "--mm-processor-cache-gb 0" in readme
    assert "--mm-processor-cache-gb 0" in deployment
    assert "--tts-latest-revision-grace-sec 4.0" in readme
    assert "--tts-latest-revision-grace-sec 4.0" in api
    assert "--tts-latest-revision-grace-sec 4.0" in deployment
    assert "不是全局 7 秒延时" in readme
    assert "source_sealed" in api
    assert "backend segment sealing" in api.lower()
    assert "voxbridge-logrotate.timer" in deployment
    assert "voxbridge-logrotate.timer" in changelog


def test_user_service_memory_containment_is_tracked_and_documented():
    dropin = (
        ROOT / "deploy" / "systemd" / "voxbridge-8024-memory.conf"
    ).read_text(encoding="utf-8")
    deployment = (ROOT / "docs" / "DEPLOYMENT.md").read_text(encoding="utf-8")

    for directive in (
        "MemoryAccounting=yes",
        "MemoryHigh=16G",
        "MemoryMax=20G",
        "TasksMax=512",
        "OOMPolicy=stop",
    ):
        assert directive in dropin
        assert directive in deployment

    assert "memory.current" in deployment
    assert "memory.peak" in deployment
    assert "memory.events" in deployment
    assert "pids.current" in deployment
    assert "disabled-systemd/voxbridge-8024-memory.conf" in deployment
    assert "external OpenAI-compatible translation service" in deployment


def test_deployment_documents_the_realtime_single_microphone_budget():
    deployment = (ROOT / "docs" / "DEPLOYMENT.md").read_text(encoding="utf-8")
    normalized = " ".join(deployment.split())

    for flag in (
        "--segment-hard-cut-sec 45",
        "--backpressure-target-queue-sec 3.0",
        "--backpressure-max-queue-sec 15.0",
        "--backpressure-hard-relief-sec 6.0",
        "--subtitle-snapshot-history-size 100",
    ):
        assert flag in deployment
    assert "never runs the blocking full-segment re-decode" in normalized
