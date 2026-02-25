import numpy as np
import pytest
import socket
from pathlib import Path

from voxbridge.cli.demo_streaming_ws import (
    INDEX_HTML_TEMPLATE,
    _acquire_instance_lock_or_raise,
    _assert_port_bindable,
    _decode_pcm16le,
    _list_orphan_enginecore_pids,
    _parse_json_message,
    _split_sentences_and_tail,
    parse_args,
)


def test_decode_pcm16le_empty():
    wav = _decode_pcm16le(b"")
    assert wav.dtype == np.float32
    assert wav.shape == (0,)


def test_decode_pcm16le_known_samples():
    raw = np.array([-32768, 0, 32767], dtype="<i2").tobytes()
    wav = _decode_pcm16le(raw)
    assert wav.shape == (3,)
    np.testing.assert_allclose(wav, np.array([-1.0, 0.0, 32767.0 / 32768.0], dtype=np.float32), rtol=0, atol=1e-7)


def test_decode_pcm16le_odd_length_raises():
    with pytest.raises(ValueError, match="even"):
        _decode_pcm16le(b"\x00")


def test_parse_json_message_accepts_object():
    payload = _parse_json_message('{"type":"finish"}')
    assert payload == {"type": "finish"}


def test_parse_json_message_rejects_invalid_json():
    with pytest.raises(ValueError, match="invalid json"):
        _parse_json_message("{")


def test_parse_json_message_rejects_non_object():
    with pytest.raises(ValueError, match="object"):
        _parse_json_message("[]")


def test_parse_args_accepts_force_language_and_max_new_tokens(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--force-language", "English", "--max-new-tokens", "48"],
    )
    args = parse_args()
    assert args.force_language == "English"
    assert args.max_new_tokens == 48


def test_parse_args_accepts_audio_queue_size(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--audio-queue-size", "12"],
    )
    args = parse_args()
    assert args.audio_queue_size == 12


def test_parse_args_accepts_consumer_batch_and_rollover(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--consumer-batch-sec", "2.5", "--state-rollover-sec", "75"],
    )
    args = parse_args()
    assert args.consumer_batch_sec == 2.5
    assert args.state_rollover_sec == 75.0


def test_parse_args_accepts_segment_and_backpressure_controls(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--segment-hard-cut-sec",
            "28",
            "--segment-overlap-sec",
            "0.9",
            "--backpressure-target-queue-sec",
            "3.2",
            "--backpressure-max-queue-sec",
            "5.7",
        ],
    )
    args = parse_args()
    assert args.segment_hard_cut_sec == 28.0
    assert args.segment_overlap_sec == 0.9
    assert args.backpressure_target_queue_sec == 3.2
    assert args.backpressure_max_queue_sec == 5.7


def test_parse_args_accepts_backend_vad_thresholds(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--backend-vad-enter-snr-db",
            "9.5",
            "--backend-vad-exit-snr-db",
            "4.8",
            "--backend-cut-stable-sec",
            "0.6",
        ],
    )
    args = parse_args()
    assert args.backend_vad_enter_snr_db == 9.5
    assert args.backend_vad_exit_snr_db == 4.8
    assert args.backend_cut_stable_sec == 0.6


def test_parse_args_accepts_auto_slice_and_overlap(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--auto-slice-sec", "25", "--slice-overlap-sec", "1.2"],
    )
    args = parse_args()
    assert args.auto_slice_sec == 25.0
    assert args.slice_overlap_sec == 1.2


def test_parse_args_accepts_finalize_on_disconnect(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--finalize-on-disconnect"],
    )
    args = parse_args()
    assert args.finalize_on_disconnect is True


def test_parse_args_accepts_subtitle_trace_options(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--subtitle-trace",
            "--subtitle-trace-max-events",
            "2400",
            "--subtitle-trace-log",
            "--subtitle-trace-log-partial-every",
            "7",
        ],
    )
    args = parse_args()
    assert args.subtitle_trace is True
    assert args.subtitle_trace_max_events == 2400
    assert args.subtitle_trace_log is True
    assert args.subtitle_trace_log_partial_every == 7


def test_index_template_contains_core_stream_controls():
    assert 'id="btnStart"' in INDEX_HTML_TEMPLATE
    assert 'id="btnStop"' in INDEX_HTML_TEMPLATE
    assert 'id="status"' in INDEX_HTML_TEMPLATE
    assert 'id="text"' in INDEX_HTML_TEMPLATE
    assert 'id="translation"' in INDEX_HTML_TEMPLATE


def test_index_template_enables_ws_backpressure_controls():
    assert "MAX_WS_BUFFERED_BYTES" in INDEX_HTML_TEMPLATE
    assert "sendQueue" in INDEX_HTML_TEMPLATE
    assert "bufferedAmount" in INDEX_HTML_TEMPLATE


def test_index_template_prefers_audio_worklet_with_fallback():
    assert "AudioWorkletNode" in INDEX_HTML_TEMPLATE
    assert "audioCtx.audioWorklet.addModule" in INDEX_HTML_TEMPLATE
    assert "createScriptProcessor" in INDEX_HTML_TEMPLATE


def test_index_template_has_audio_watchdog_markers():
    assert "No audio input / 未检测到音频输入" in INDEX_HTML_TEMPLATE
    assert "startWatchdog" in INDEX_HTML_TEMPLATE


def test_index_template_removes_frontend_slice_and_vad_markers():
    assert "AUTO_SLICE_SEC" not in INDEX_HTML_TEMPLATE
    assert "SLICE_OVERLAP_SEC" not in INDEX_HTML_TEMPLATE
    assert "VAD_SILENCE_SEC" not in INDEX_HTML_TEMPLATE
    assert "VAD_MIN_SLICE_SEC" not in INDEX_HTML_TEMPLATE
    assert "VAD_MIN_ACTIVE_SEC" not in INDEX_HTML_TEMPLATE
    assert "VAD_FORCE_CUT_SEC" not in INDEX_HTML_TEMPLATE
    assert "rotateSliceSession" not in INDEX_HTML_TEMPLATE


def test_index_template_disables_frontend_slice_state_machine():
    assert 'const SLICE_MODE = "off";' not in INDEX_HTML_TEMPLATE
    assert 'await sendFinishAndAwaitFinal("slice"' not in INDEX_HTML_TEMPLATE
    assert 'if (mode === "slice") {' not in INDEX_HTML_TEMPLATE


def test_index_template_uses_two_thirds_height_for_english_lane():
    assert "grid-template-rows: 2fr 1fr;" in INDEX_HTML_TEMPLATE


def test_index_template_prefers_committed_sentence_events_for_stream_ui():
    assert "const USE_COMMITTED_SENTENCE_EVENTS = true;" in INDEX_HTML_TEMPLATE
    assert "const tentativeTail = String(msg.tentative_text || \"\").trim();" in INDEX_HTML_TEMPLATE
    assert "if (USE_COMMITTED_SENTENCE_EVENTS) {" in INDEX_HTML_TEMPLATE
    assert "const reserveTailSlot = running && committedRows.length > 0;" in INDEX_HTML_TEMPLATE
    assert "return clipVisibleRows(head);" in INDEX_HTML_TEMPLATE


def test_index_template_limits_visible_rows_to_four():
    assert "const MAX_VISIBLE_ROWS = 4;" in INDEX_HTML_TEMPLATE
    assert "rows.slice(rows.length - MAX_VISIBLE_ROWS)" in INDEX_HTML_TEMPLATE


def test_index_template_clips_rows_after_sentence_cap_in_committed_mode():
    assert "return clipVisibleRows(head);" in INDEX_HTML_TEMPLATE


def test_index_template_clears_dom_before_resetting_line_node_maps():
    assert "function clearSubtitleDom(){" in INDEX_HTML_TEMPLATE
    assert "if (textEl) textEl.replaceChildren();" in INDEX_HTML_TEMPLATE
    assert "if (translationEl) translationEl.replaceChildren();" in INDEX_HTML_TEMPLATE
    assert "clearSubtitleDom();" in INDEX_HTML_TEMPLATE


def test_index_template_contains_subtitle_trace_hooks():
    assert "const SUBTITLE_TRACE_DEFAULT = __SUBTITLE_TRACE__;" in INDEX_HTML_TEMPLATE
    assert "const SUBTITLE_TRACE_MAX_EVENTS = __SUBTITLE_TRACE_MAX_EVENTS__;" in INDEX_HTML_TEMPLATE
    assert "function traceSubtitle" in INDEX_HTML_TEMPLATE
    assert "window.__subtitleDebug" in INDEX_HTML_TEMPLATE
    assert "getTrace(limit)" in INDEX_HTML_TEMPLATE
    assert "setTraceEnabled(enabled)" in INDEX_HTML_TEMPLATE


def test_index_template_supports_safe_sentence_updated_overwrite():
    assert "function isCommittedSentenceUpgrade" not in INDEX_HTML_TEMPLATE
    assert "function reconcileNextSentenceAfterOverwrite" not in INDEX_HTML_TEMPLATE
    assert "const allowOverwrite = true;" in INDEX_HTML_TEMPLATE
    assert "{ allowOverwrite, sliceCommit: !!msg.slice_commit }" in INDEX_HTML_TEMPLATE


def test_index_template_handles_sentence_updated_event():
    assert 'msg.type === "sentence_updated"' in INDEX_HTML_TEMPLATE


def test_index_template_has_no_frontend_text_based_row_splitters():
    assert "const MAX_SENTENCES_PER_ROW = 1;" not in INDEX_HTML_TEMPLATE
    assert "function splitTextByDisplayRules" not in INDEX_HTML_TEMPLATE
    assert "function alignTranslationChunks" not in INDEX_HTML_TEMPLATE
    assert "function splitRowBySentenceCap" not in INDEX_HTML_TEMPLATE
    assert "applySentenceCap(rows)" not in INDEX_HTML_TEMPLATE
    assert "function mergeSliceCommittedRows" not in INDEX_HTML_TEMPLATE
    assert "function normalizeSubtitleRows" not in INDEX_HTML_TEMPLATE


def test_index_template_has_no_legacy_text_inference_pipeline():
    assert "function extractTailByCommitted" not in INDEX_HTML_TEMPLATE
    assert "function longestCommonPrefixLen" not in INDEX_HTML_TEMPLATE
    assert "function stripBoundaryOverlap" not in INDEX_HTML_TEMPLATE
    assert "function mergeTranscript" not in INDEX_HTML_TEMPLATE
    assert "function rebuildSubtitleWindow" not in INDEX_HTML_TEMPLATE
    assert 'msg.type === "translation"' not in INDEX_HTML_TEMPLATE


def test_index_template_avoids_committed_row_rewrite_and_empty_translation_reset():
    assert "{ allowOverwrite: false, sliceCommit: !!msg.slice_commit }" not in INDEX_HTML_TEMPLATE
    assert "if (!enText) {" in INDEX_HTML_TEMPLATE
    assert "if (cur) return;" not in INDEX_HTML_TEMPLATE
    assert "translation_updated_local" in INDEX_HTML_TEMPLATE


def test_index_template_stabilizes_tentative_tail_to_avoid_flash_drop():
    assert "const TAIL_STABILIZE_MS = 700;" in INDEX_HTML_TEMPLATE
    assert "function updateCommittedTentativeTail" in INDEX_HTML_TEMPLATE
    assert "tailStabilizeTimer = setTimeout" in INDEX_HTML_TEMPLATE
    assert "const reserveTailSlot = running && committedRows.length > 0;" in INDEX_HTML_TEMPLATE


def test_index_template_uses_stop_only_finish_mode():
    assert "const payload = {type: \"finish\", mode};" in INDEX_HTML_TEMPLATE
    assert "if (reason) payload.reason = String(reason);" in INDEX_HTML_TEMPLATE
    assert "ws.send(JSON.stringify(payload));" in INDEX_HTML_TEMPLATE
    assert 'await sendFinishAndAwaitFinal("stop", 45000);' in INDEX_HTML_TEMPLATE
    assert 'sock.send(JSON.stringify({type: "finish", mode: "stop"}));' in INDEX_HTML_TEMPLATE


def test_index_template_blocks_start_reentry_while_awaiting_final():
    assert "if (running || awaitingFinal) return;" in INDEX_HTML_TEMPLATE
    assert "awaitingFinal = true;" in INDEX_HTML_TEMPLATE
    assert 'setStatus("Finishing / 收尾中", "warn");' in INDEX_HTML_TEMPLATE
    assert 'setStatus("Stopped / 已停止", "");' in INDEX_HTML_TEMPLATE
    assert "lockUI(false);" in INDEX_HTML_TEMPLATE


def test_index_template_single_stream_state_no_frontend_slice_reopen():
    assert "async function rotateSliceSession(reason = \"time\"){" not in INDEX_HTML_TEMPLATE
    assert "await openSocket();" in INDEX_HTML_TEMPLATE
    assert 'type: "start"' in INDEX_HTML_TEMPLATE
    assert "language: selectedLanguage()" in INDEX_HTML_TEMPLATE
    assert "translation_direction: selectedTranslationDirection()" in INDEX_HTML_TEMPLATE


def test_backend_final_commit_uses_stop_mode_tail_flush():
    src = Path(__file__).resolve().parents[1] / "voxbridge" / "cli" / "demo_streaming_ws.py"
    text = src.read_text(encoding="utf-8")
    assert "force_tail=True" in text
    assert "holdback_newest=False" in text
    assert "commit_tail_if_no_completed=False" in text
    assert "commit_tail_always=False" in text
    assert "commit_all_completed=False" in text
    assert "slice_commit=False" in text


def test_backend_final_commit_no_slice_branch_left():
    src = Path(__file__).resolve().parents[1] / "voxbridge" / "cli" / "demo_streaming_ws.py"
    text = src.read_text(encoding="utf-8")
    assert "holdback_newest: bool = True" in text
    assert "force_slice_tail_guard = bool(finish_mode == \"slice\" and finish_reason == \"force\")" not in text
    assert "slice_final = bool(finish_mode == \"slice\")" not in text


def test_frontend_has_no_vad_state_machine_artifacts():
    assert "function hasSliceBoundary(text){" not in INDEX_HTML_TEMPLATE
    assert "VAD_FORCE_CUT_EXTRA_MS" not in INDEX_HTML_TEMPLATE
    assert "VAD_SPEECH_CONFIRM_MS" not in INDEX_HTML_TEMPLATE
    assert "resetVadState" not in INDEX_HTML_TEMPLATE
    assert "vad_slice_trigger_idle_text" not in INDEX_HTML_TEMPLATE


def test_index_route_no_longer_replaces_removed_frontend_slice_placeholders():
    src = Path(__file__).resolve().parents[1] / "voxbridge" / "cli" / "demo_streaming_ws.py"
    text = src.read_text(encoding="utf-8")
    assert 'html = html.replace("__SLICE_MODE__"' not in text
    assert 'html = html.replace("__AUTO_SLICE_SEC__"' not in text
    assert 'html = html.replace("__SLICE_OVERLAP_SEC__"' not in text
    assert 'html = html.replace("__VAD_SILENCE_SEC__"' not in text
    assert 'html = html.replace("__VAD_MIN_SLICE_SEC__"' not in text
    assert 'html = html.replace("__VAD_MIN_ACTIVE_SEC__"' not in text
    assert 'html = html.replace("__VAD_FORCE_CUT_SEC__"' not in text


def test_backend_ignores_slice_finish_mode_from_client():
    src = Path(__file__).resolve().parents[1] / "voxbridge" / "cli" / "demo_streaming_ws.py"
    text = src.read_text(encoding="utf-8")
    assert "finish_reason = \"stop\"" in text
    assert "requested_reason = str(payload.get(\"reason\", \"\") or \"\").strip().lower()" in text
    assert "if requested_mode == \"slice\":" in text
    assert "_trace_event(\"finish_slice_ignored\", requested_reason=requested_reason)" in text


def test_backend_uses_single_state_per_ws_and_stops_on_finish():
    src = Path(__file__).resolve().parents[1] / "voxbridge" / "cli" / "demo_streaming_ws.py"
    text = src.read_text(encoding="utf-8")
    assert "finish_requested = False" in text
    assert "finished = False" in text
    assert "if consumer_task is None or consumer_task.done():" in text
    assert "consumer_task = asyncio.create_task(_audio_consumer())" in text
    assert "if finish_mode == \"slice\":" not in text
    assert "break" in text


def test_backend_partial_final_emit_incremental_delta_fields():
    src = Path(__file__).resolve().parents[1] / "voxbridge" / "cli" / "demo_streaming_ws.py"
    text = src.read_text(encoding="utf-8")
    assert "def _compute_text_delta(" in text
    assert "payload[\"delta_text\"] = delta_text" in text
    assert "payload[\"text_reset\"] = bool(text_reset)" in text
    assert "payload[\"state_text\"] = full_text" in text


def test_backend_text_pool_trace_has_generating_and_solidified_phase():
    src = Path(__file__).resolve().parents[1] / "voxbridge" / "cli" / "demo_streaming_ws.py"
    text = src.read_text(encoding="utf-8")
    assert "\"topic\": \"text_pool\"" in text
    assert "\"phase\": str(phase or \"\")" in text
    assert "\"segment_id\": int(getattr(segment_runtime, \"id\", 0) or 0)" in text
    assert "\"text_hash8\": _hash8(snapshot)" in text
    assert "pool_generating_set" in text
    assert "pool_solidified_append" in text


def test_backend_forces_finish_mode_stop_for_single_stream_state():
    src = Path(__file__).resolve().parents[1] / "voxbridge" / "cli" / "demo_streaming_ws.py"
    text = src.read_text(encoding="utf-8")
    assert "finish_mode = \"stop\"" in text
    assert "finish_reason = \"stop\"" in text
    assert "finish_mode = \"slice\" if requested_mode == \"slice\" else \"stop\"" not in text
    assert "if finish_mode == \"slice\":" not in text


def test_backend_has_idle_tail_commit_fallback_for_translation():
    src = Path(__file__).resolve().parents[1] / "voxbridge" / "cli" / "demo_streaming_ws.py"
    text = src.read_text(encoding="utf-8")
    assert "idle_commit_sec = max(3.0, float(getattr(args, \"vad_force_cut_sec\", 1.8)) + 2.7)" in text
    assert "async def _maybe_idle_tail_commit() -> None:" in text
    assert "commit_tail_always=allow_tail_commit," in text
    assert "allow_tail_commit = bool(tail_looks_complete or tail_meets_min_len)" in text
    assert "_trace_event(" in text and "idle_tail_commit" in text
    assert "await _maybe_idle_tail_commit()" in text


def test_index_template_updates_tail_directly_from_backend_tentative_text():
    assert "const bridgeCandidate = tentativeTail || String(nextText || \"\").trim();" not in INDEX_HTML_TEMPLATE
    assert "if (holdBoundaryTail) {" not in INDEX_HTML_TEMPLATE
    assert "updateCommittedTentativeTail(tentativeTail);" in INDEX_HTML_TEMPLATE
    assert "window.__subtitleDebug" in INDEX_HTML_TEMPLATE


def test_index_template_avoids_unconditional_merge_for_adjacent_slice_commits():
    assert "if (prevSlice && curSlice) return true;" not in INDEX_HTML_TEMPLATE
    assert "function mergeSliceCommittedRows" not in INDEX_HTML_TEMPLATE


def test_split_sentences_merges_short_cjk_fragment_before_long_sentence():
    text = "他就离。”他父亲死了以后，神使他从那里搬到你们现在所住之地。"
    sentences, tail = _split_sentences_and_tail(text)
    assert tail == ""
    assert len(sentences) == 1
    assert "他就离。" in sentences[0]
    assert "他父亲死了以后" in sentences[0]


def test_split_sentences_avoids_tiny_cjk_sentences_in_long_quote():
    text = (
        "大祭司就说：“这些事果然有吗？”史提凡说：“诸位父兄，请听，当日我们的祖宗亚伯拉罕"
        "在美索不达米亚还未住哈兰的时候，荣耀的神向他显现，对他说：‘你要离开本地和亲族，往我所要"
        "指示你的地方去。’他就离开迦勒底人之地，住在哈兰。他父亲死了以后，神使他从那里搬到你们现在"
        "所住之地。在这地方，神并没有给他产业，连立足之地也没有给他，但应许要要将这一块地赐给他和他"
        "的后裔为业。那时他还没有儿子。神说：他的后裔必寄居外邦。”"
    )
    sentences, tail = _split_sentences_and_tail(text)
    assert tail == ""
    cjk_sentences = [s for s in sentences if any("\u4e00" <= ch <= "\u9fff" for ch in s)]
    assert cjk_sentences
    assert all(len(s) >= 10 for s in cjk_sentences)


def test_split_sentences_keeps_original_punctuation_without_style_rewrite():
    sentences, tail = _split_sentences_and_tail("第三次测试翻译，第四次测试翻译，第五。次测试翻译，第六次测试翻译。")
    assert tail == ""
    joined = "".join(sentences)
    assert "第五。次测试翻译" in joined


def test_index_template_hides_raw_asr_text_panel():
    assert "Raw ASR Text" not in INDEX_HTML_TEMPLATE
    assert 'id="rawText"' not in INDEX_HTML_TEMPLATE
    assert "class=\"raw-panel\"" not in INDEX_HTML_TEMPLATE


def test_index_template_has_translation_direction_toggle_and_ws_control():
    assert 'id="translationDirectionToggle"' in INDEX_HTML_TEMPLATE
    assert 'id="translationDirectionLabel"' in INDEX_HTML_TEMPLATE
    assert "function selectedTranslationDirection()" in INDEX_HTML_TEMPLATE
    assert 'type: "set_translation_direction"' in INDEX_HTML_TEMPLATE
    assert "translation_direction: selectedTranslationDirection()" in INDEX_HTML_TEMPLATE


def test_index_template_has_audio_input_source_selector():
    assert 'id="inputSourceSelect"' in INDEX_HTML_TEMPLATE
    assert 'id="inputSourceLabel"' in INDEX_HTML_TEMPLATE
    assert 'value="mic"' in INDEX_HTML_TEMPLATE
    assert 'value="system"' in INDEX_HTML_TEMPLATE
    assert "function selectedInputSource()" in INDEX_HTML_TEMPLATE


def test_index_template_supports_system_audio_capture_via_display_media():
    assert "function openSystemAudio()" in INDEX_HTML_TEMPLATE
    assert "navigator.mediaDevices.getDisplayMedia" in INDEX_HTML_TEMPLATE
    assert "displaySurface: \"monitor\"" in INDEX_HTML_TEMPLATE
    assert "systemAudio: \"include\"" in INDEX_HTML_TEMPLATE
    assert "preferCurrentTab: false" in INDEX_HTML_TEMPLATE
    assert "selfBrowserSurface: \"exclude\"" in INDEX_HTML_TEMPLATE
    assert "if (!audioTracks || audioTracks.length === 0)" in INDEX_HTML_TEMPLATE
    assert "请选择整屏共享并勾选系统音频" in INDEX_HTML_TEMPLATE


def test_port_precheck_rejects_occupied_port():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    probe.listen(1)
    port = int(probe.getsockname()[1])
    try:
        with pytest.raises(RuntimeError, match="not available"):
            _assert_port_bindable("127.0.0.1", port)
    finally:
        probe.close()


def test_instance_lock_rejects_second_holder(tmp_path):
    lock_path = tmp_path / "streaming_8024.lock"
    handle = _acquire_instance_lock_or_raise(8024, lock_path=lock_path)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            _acquire_instance_lock_or_raise(8024, lock_path=lock_path)
    finally:
        handle.close()


def test_list_orphan_enginecore_pids_filters_ppid_and_uid(tmp_path):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()

    def write_status(pid: int, *, name: str, ppid: int, uid: int) -> None:
        p = proc_root / str(pid)
        p.mkdir()
        (p / "status").write_text(
            (
                f"Name:\t{name}\n"
                f"State:\tS (sleeping)\n"
                f"PPid:\t{ppid}\n"
                f"Uid:\t{uid}\t{uid}\t{uid}\t{uid}\n"
            ),
            encoding="utf-8",
        )

    write_status(101, name="VLLM::EngineCor", ppid=1, uid=1000)   # keep
    write_status(102, name="VLLM::EngineCor", ppid=999, uid=1000) # no: ppid
    write_status(103, name="VLLM::EngineCor", ppid=1, uid=1001)   # no: uid
    write_status(104, name="python", ppid=1, uid=1000)            # no: name
    write_status(105, name="VLLM::EngineCore", ppid=1, uid=1000)  # keep

    got = _list_orphan_enginecore_pids(proc_root=proc_root, current_uid=1000)
    assert got == [101, 105]
