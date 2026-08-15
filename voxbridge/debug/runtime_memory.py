from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class DRMMemorySnapshot:
    gtt_bytes: int
    vram_bytes: int
    cpu_bytes: int
    client_count: int


@dataclass(frozen=True)
class ProcessMemorySnapshot:
    pid: int
    comm: str
    rss_bytes: int | None
    pss_bytes: int | None
    pss_anon_bytes: int | None
    pss_file_bytes: int | None
    pss_shmem_bytes: int | None
    threads: int | None
    fd_count: int | None
    gtt_bytes: int
    vram_bytes: int
    drm_client_count: int


@dataclass(frozen=True)
class RuntimeMemorySnapshot:
    captured_at_ms: int
    cgroup_path: str
    cgroup_memory_current_bytes: int | None
    cgroup_memory_peak_bytes: int | None
    cgroup_events: Mapping[str, int]
    cgroup_task_count: int | None
    processes: tuple[ProcessMemorySnapshot, ...]
    enginecore_count: int
    main_pid: int | None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None


def _read_int(path: Path) -> int | None:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return int(text.strip())
    except ValueError:
        return None


def _parse_key_values(text: str | None) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in str(text or "").splitlines():
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip()] = value.strip()
    return values


def _parse_kib(value: str | None) -> int | None:
    if value is None:
        return None
    fields = value.split()
    if not fields:
        return None
    try:
        amount = int(fields[0])
    except ValueError:
        return None
    unit = fields[1].lower() if len(fields) > 1 else "kib"
    multipliers = {"b": 1, "kb": 1024, "kib": 1024, "mb": 1024**2, "mib": 1024**2}
    multiplier = multipliers.get(unit)
    return None if multiplier is None else amount * multiplier


def read_drm_memory_bytes(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> DRMMemorySnapshot:
    fdinfo_root = Path(proc_root) / str(int(pid)) / "fdinfo"
    clients: dict[str, tuple[int, int, int]] = {}
    try:
        entries = tuple(fdinfo_root.iterdir())
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        entries = ()
    for entry in entries:
        values = _parse_key_values(_read_text(entry))
        client_id = values.get("drm-client-id")
        if not client_id:
            continue
        current = (
            _parse_kib(values.get("drm-memory-gtt")) or 0,
            _parse_kib(values.get("drm-memory-vram")) or 0,
            _parse_kib(values.get("drm-memory-cpu")) or 0,
        )
        previous = clients.get(client_id, (0, 0, 0))
        clients[client_id] = tuple(max(old, new) for old, new in zip(previous, current))
    return DRMMemorySnapshot(
        gtt_bytes=sum(item[0] for item in clients.values()),
        vram_bytes=sum(item[1] for item in clients.values()),
        cpu_bytes=sum(item[2] for item in clients.values()),
        client_count=len(clients),
    )


def read_process_memory(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> ProcessMemorySnapshot | None:
    root = Path(proc_root) / str(int(pid))
    status_text = _read_text(root / "status")
    comm_text = _read_text(root / "comm")
    if status_text is None or comm_text is None:
        return None
    status = _parse_key_values(status_text)
    smaps = _parse_key_values(_read_text(root / "smaps_rollup"))
    drm = read_drm_memory_bytes(pid, proc_root=proc_root)
    try:
        fd_count: int | None = sum(1 for _ in (root / "fd").iterdir())
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        fd_count = None
    try:
        threads = int(status["Threads"].split()[0])
    except (KeyError, ValueError, IndexError):
        threads = None
    return ProcessMemorySnapshot(
        pid=int(pid),
        comm=comm_text.strip(),
        rss_bytes=_parse_kib(status.get("VmRSS")),
        pss_bytes=_parse_kib(smaps.get("Pss")),
        pss_anon_bytes=_parse_kib(smaps.get("Pss_Anon")),
        pss_file_bytes=_parse_kib(smaps.get("Pss_File")),
        pss_shmem_bytes=_parse_kib(smaps.get("Pss_Shmem")),
        threads=threads,
        fd_count=fd_count,
        gtt_bytes=drm.gtt_bytes,
        vram_bytes=drm.vram_bytes,
        drm_client_count=drm.client_count,
    )


def _read_events(path: Path) -> dict[str, int]:
    events: dict[str, int] = {}
    for line in str(_read_text(path) or "").splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            events[fields[0]] = int(fields[1])
        except ValueError:
            continue
    return events


def _is_enginecore_comm(comm: str) -> bool:
    # Linux task names are limited to 15 visible bytes in /proc/<pid>/comm.
    return comm in {"VLLM::EngineCore", "VLLM::EngineCor"}


def read_runtime_memory(
    cgroup_path: Path,
    *,
    proc_root: Path = Path("/proc"),
) -> RuntimeMemorySnapshot:
    cgroup = Path(cgroup_path)
    procs_text = _read_text(cgroup / "cgroup.procs")
    if procs_text is None:
        raise FileNotFoundError(f"cgroup is unavailable: {cgroup}")
    pids: list[int] = []
    for field in procs_text.split():
        try:
            pids.append(int(field))
        except ValueError:
            continue
    processes = tuple(
        process
        for pid in sorted(set(pids))
        if (process := read_process_memory(pid, proc_root=proc_root)) is not None
    )
    return RuntimeMemorySnapshot(
        captured_at_ms=int(time.time() * 1000),
        cgroup_path=str(cgroup),
        cgroup_memory_current_bytes=_read_int(cgroup / "memory.current"),
        cgroup_memory_peak_bytes=_read_int(cgroup / "memory.peak"),
        cgroup_events=_read_events(cgroup / "memory.events"),
        cgroup_task_count=_read_int(cgroup / "pids.current"),
        processes=processes,
        enginecore_count=sum(_is_enginecore_comm(process.comm) for process in processes),
        main_pid=min((process.pid for process in processes), default=None),
    )


__all__ = [
    "DRMMemorySnapshot",
    "ProcessMemorySnapshot",
    "RuntimeMemorySnapshot",
    "read_drm_memory_bytes",
    "read_process_memory",
    "read_runtime_memory",
]
