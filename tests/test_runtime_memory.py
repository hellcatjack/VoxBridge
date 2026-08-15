from __future__ import annotations

from pathlib import Path

from voxbridge.debug.runtime_memory import (
    read_drm_memory_bytes,
    read_process_memory,
    read_runtime_memory,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_process(
    proc_root: Path,
    pid: int,
    *,
    comm: str,
    rss_kib: int,
    pss_kib: int,
    threads: int,
) -> None:
    root = proc_root / str(pid)
    _write(
        root / "status",
        f"Name:\t{comm}\nVmRSS:\t{rss_kib} kB\nThreads:\t{threads}\n",
    )
    _write(
        root / "smaps_rollup",
        "\n".join(
            (
                f"Rss: {rss_kib} kB",
                f"Pss: {pss_kib} kB",
                f"Pss_Anon: {pss_kib - 30} kB",
                "Pss_File: 20 kB",
                "Pss_Shmem: 10 kB",
            )
        )
        + "\n",
    )
    _write(root / "comm", f"{comm}\n")
    (root / "fd").mkdir(parents=True)


def test_drm_memory_deduplicates_multiple_fds_for_one_client(tmp_path):
    proc_root = tmp_path / "proc"
    _make_process(
        proc_root,
        42,
        comm="VLLM::EngineCore",
        rss_kib=300,
        pss_kib=250,
        threads=10,
    )
    fdinfo = proc_root / "42" / "fdinfo"
    duplicate = (
        "drm-client-id:\t7\n"
        "drm-memory-vram:\t128 KiB\n"
        "drm-memory-gtt: \t4096 KiB\n"
    )
    _write(fdinfo / "8", duplicate)
    _write(fdinfo / "9", duplicate)
    _write(
        fdinfo / "10",
        "drm-client-id:\t8\ndrm-memory-vram:\t64 KiB\n"
        "drm-memory-gtt: \t1024 KiB\n",
    )

    memory = read_drm_memory_bytes(42, proc_root=proc_root)

    assert memory.gtt_bytes == (4096 + 1024) * 1024
    assert memory.vram_bytes == (128 + 64) * 1024
    assert memory.client_count == 2


def test_process_memory_reads_pss_rss_threads_and_fds(tmp_path):
    proc_root = tmp_path / "proc"
    _make_process(
        proc_root,
        100,
        comm="python",
        rss_kib=1024,
        pss_kib=900,
        threads=17,
    )
    _write(
        proc_root / "100" / "fdinfo" / "7",
        "drm-client-id:\t12\ndrm-memory-gtt: \t2048 KiB\n",
    )
    for name in ("0", "1", "7"):
        (proc_root / "100" / "fd" / name).touch()

    snapshot = read_process_memory(100, proc_root=proc_root)

    assert snapshot is not None
    assert snapshot.pid == 100
    assert snapshot.comm == "python"
    assert snapshot.rss_bytes == 1024 * 1024
    assert snapshot.pss_bytes == 900 * 1024
    assert snapshot.pss_anon_bytes == 870 * 1024
    assert snapshot.pss_file_bytes == 20 * 1024
    assert snapshot.pss_shmem_bytes == 10 * 1024
    assert snapshot.threads == 17
    assert snapshot.fd_count == 3
    assert snapshot.gtt_bytes == 2048 * 1024


def test_process_memory_returns_none_when_process_disappears(tmp_path):
    assert read_process_memory(999, proc_root=tmp_path / "proc") is None


def test_runtime_memory_enumerates_only_cgroup_processes(tmp_path):
    proc_root = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    _make_process(
        proc_root,
        201,
        comm="python",
        rss_kib=100,
        pss_kib=80,
        threads=4,
    )
    _make_process(
        proc_root,
        202,
        comm="VLLM::EngineCore",
        rss_kib=200,
        pss_kib=160,
        threads=8,
    )
    _make_process(
        proc_root,
        999,
        comm="VLLM::EngineCore",
        rss_kib=999,
        pss_kib=999,
        threads=99,
    )
    _write(cgroup / "cgroup.procs", "201\n202\n303\n")
    _write(cgroup / "memory.current", "123456\n")
    _write(cgroup / "memory.peak", "234567\n")
    _write(cgroup / "memory.events", "low 1\nhigh 2\nmax 3\noom 0\noom_kill 0\n")
    _write(cgroup / "pids.current", "12\n")

    snapshot = read_runtime_memory(cgroup, proc_root=proc_root)

    assert snapshot.cgroup_memory_current_bytes == 123456
    assert snapshot.cgroup_memory_peak_bytes == 234567
    assert snapshot.cgroup_events == {
        "low": 1,
        "high": 2,
        "max": 3,
        "oom": 0,
        "oom_kill": 0,
    }
    assert snapshot.cgroup_task_count == 12
    assert [process.pid for process in snapshot.processes] == [201, 202]
    assert snapshot.enginecore_count == 1
    assert snapshot.main_pid == 201


def test_runtime_memory_recognizes_linux_truncated_enginecore_comm(tmp_path):
    proc_root = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    _make_process(
        proc_root,
        202,
        comm="VLLM::EngineCor",
        rss_kib=200,
        pss_kib=160,
        threads=8,
    )
    _write(cgroup / "cgroup.procs", "202\n")

    snapshot = read_runtime_memory(cgroup, proc_root=proc_root)

    assert snapshot.enginecore_count == 1
