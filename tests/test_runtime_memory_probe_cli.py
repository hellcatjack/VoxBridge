from __future__ import annotations

import json
from pathlib import Path

from tools import runtime_memory_probe


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _runtime_fixture(tmp_path: Path) -> tuple[Path, Path]:
    cgroup = tmp_path / "cgroup"
    proc_root = tmp_path / "proc"
    _write(cgroup / "cgroup.procs", "81\n")
    _write(cgroup / "memory.current", "4096\n")
    _write(cgroup / "memory.peak", "8192\n")
    _write(cgroup / "memory.events", "oom 0\noom_kill 0\n")
    _write(cgroup / "pids.current", "3\n")
    _write(proc_root / "81" / "comm", "VLLM::EngineCore\n")
    _write(proc_root / "81" / "status", "VmRSS:\t10 kB\nThreads:\t2\n")
    _write(proc_root / "81" / "smaps_rollup", "Pss: 8 kB\n")
    (proc_root / "81" / "fd").mkdir(parents=True)
    _write(
        proc_root / "81" / "fdinfo" / "8",
        "drm-client-id:\t1\ndrm-memory-gtt:\t20 KiB\n",
    )
    return cgroup, proc_root


def test_probe_writes_one_flushed_jsonl_sample(tmp_path):
    cgroup, proc_root = _runtime_fixture(tmp_path)
    output = tmp_path / "samples.jsonl"

    exit_code = runtime_memory_probe.main(
        [
            "--cgroup",
            str(cgroup),
            "--proc-root",
            str(proc_root),
            "--output",
            str(output),
            "--sample-count",
            "1",
            "--interval-sec",
            "0",
        ]
    )

    assert exit_code == 0
    raw = output.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 1
    assert rows[0]["elapsed_ms"] >= 0
    assert rows[0]["runtime"]["cgroup_memory_current_bytes"] == 4096
    assert rows[0]["runtime"]["enginecore_count"] == 1
    assert rows[0]["runtime"]["processes"][0]["gtt_bytes"] == 20 * 1024


def test_probe_appends_samples_without_rewriting_history(tmp_path):
    cgroup, proc_root = _runtime_fixture(tmp_path)
    output = tmp_path / "samples.jsonl"
    output.write_text('{"existing":true}\n', encoding="utf-8")

    exit_code = runtime_memory_probe.main(
        [
            "--cgroup",
            str(cgroup),
            "--proc-root",
            str(proc_root),
            "--output",
            str(output),
            "--sample-count",
            "2",
            "--interval-sec",
            "0",
        ]
    )

    assert exit_code == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0] == {"existing": True}
    assert len(rows) == 3


def test_probe_returns_nonzero_for_missing_cgroup(tmp_path, capsys):
    exit_code = runtime_memory_probe.main(
        [
            "--cgroup",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "samples.jsonl"),
            "--sample-count",
            "1",
        ]
    )

    assert exit_code == 2
    assert "cgroup is unavailable" in capsys.readouterr().err


def test_probe_binds_health_and_tts_payload_to_each_sample(tmp_path, monkeypatch):
    cgroup, proc_root = _runtime_fixture(tmp_path)
    output = tmp_path / "samples.jsonl"

    def fake_fetch(url: str, timeout_sec: float):
        if url.endswith("/status"):
            return 200, {"listener_count": 1, "queue_depth": 2, "ignored": "value"}, ""
        return 200, None, ""

    monkeypatch.setattr(runtime_memory_probe, "fetch_http", fake_fetch)
    exit_code = runtime_memory_probe.main(
        [
            "--cgroup",
            str(cgroup),
            "--proc-root",
            str(proc_root),
            "--output",
            str(output),
            "--sample-count",
            "1",
            "--health-url",
            "http://127.0.0.1:8024/listen",
            "--tts-status-url",
            "http://127.0.0.1:8024/status",
        ]
    )

    assert exit_code == 0
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["health"] == {"status": 200, "error": ""}
    assert row["tts_status"] == {
        "status": 200,
        "error": "",
        "listener_count": 1,
        "queue_depth": 2,
    }
