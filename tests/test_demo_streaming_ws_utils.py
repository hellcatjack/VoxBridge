import asyncio
import numpy as np
import pytest
import socket
import re
import json
import threading
import urllib.request
from pathlib import Path

import voxbridge.cli.demo_streaming_ws as demo_streaming_ws
from voxbridge.cli.demo_streaming_ws import (
    INDEX_HTML_TEMPLATE,
    OpenAIAPITranslator,
    _await_thread_completion_on_cancel,
    _acquire_instance_lock_or_raise,
    _alignment_registry_touch,
    _alignment_registry_touch_model,
    _summarize_alignment_gap,
    _assert_port_bindable,
    _should_accept_sentence_upgrade,
    _is_probable_pending_prefix_duplicate,
    _should_hold_partial_reset,
    _should_release_partial_reset_guard,
    _should_apply_carry_overlap_skip,
    _decode_pcm16le,
    _is_short_english_sentence_for_early_commit,
    _looks_like_asr_context_echo,
    _filter_asr_context_echo_sentences,
    _safe_exception_trace_fields,
    _should_accept_context_sentence_correction,
    _list_orphan_enginecore_pids,
    _parse_json_message,
    _should_skip_stream_decode,
    _should_use_high_batch_merge,
    _split_sentences_and_tail,
    _find_first_boundary_after,
    _hash_auth_password,
    _trim_leading_boundary_overlap,
    _verify_auth_password,
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


def test_context_echo_guard_rejects_glossary_copy_but_not_natural_speech():
    context = "流便 扫罗 迦南女子 暗兰 约基别 亚伦 摩西 近亲婚姻"

    assert _looks_like_asr_context_echo(context, context + "。") is True
    assert _looks_like_asr_context_echo(context, "流便和扫罗都出现在这一段家谱中。") is False
    assert _looks_like_asr_context_echo("出埃及记", "出埃及记。") is False
    assert _looks_like_asr_context_echo(
        "Amram Moses Aaron",
        "Amram, Moses, and Aaron.",
        previous_text="Amron, Moses, and Aaron.",
    ) is False


def test_context_echo_filter_removes_only_glossary_like_sentences():
    context = "Reuben Saul Canaanite Amram Jochebed Aaron Moses"
    glossary_copy = "Reuben Saul Canaanite Amram Jochebed Aaron Moses."
    source = (
        "The genealogy introduces several families. "
        f"{glossary_copy} "
        "Moses later returned to Egypt."
    )

    filtered, removed = _filter_asr_context_echo_sentences(context, source)

    assert removed == 1
    assert glossary_copy not in filtered
    assert "The genealogy introduces several families." in filtered
    assert "Moses later returned to Egypt." in filtered

    natural = "Reuben and Saul both appear in this part of the genealogy."
    assert _filter_asr_context_echo_sentences(context, natural) == (natural, 0)
    assert _filter_asr_context_echo_sentences("Exodus", "Exodus.") == ("Exodus.", 0)


def test_context_echo_filter_preserves_a_spoken_glossary_with_incremental_evidence():
    context = "Reuben Saul Canaanite Amram Jochebed Aaron Moses"
    prefix = (
        "The reading begins here with a long explanation of the historical setting. "
        "Another introductory sentence gives the audience additional background."
    )
    previous = prefix + " Reuben Saul Canaanite Amram Jochebed"
    spoken = prefix + " " + context + "."

    filtered, removed = _filter_asr_context_echo_sentences(
        context,
        spoken,
        previous_text=previous,
    )

    assert removed == 0
    assert context in filtered


def test_compact_occurrence_coverage_consumes_each_existing_span_once():
    existing = "thefamilyrecordissimpleanditcontainsmanyancestralnames"
    used_spans = []

    assert demo_streaming_ws._consume_unmatched_compact_occurrence(
        existing,
        "thefamilyrecordissimple",
        used_spans,
    ) is True
    assert demo_streaming_ws._consume_unmatched_compact_occurrence(
        existing,
        "itcontainsmanyancestralnames",
        used_spans,
    ) is True
    assert demo_streaming_ws._consume_unmatched_compact_occurrence(
        existing,
        "itcontainsmanyancestralnames",
        used_spans,
    ) is False


def test_context_sentence_correction_accepts_lexical_revision_only_when_aligned():
    assert _should_accept_context_sentence_correction(
        "Amron went home.",
        "Amram went home.",
    ) is True
    assert _should_accept_context_sentence_correction(
        "摩西和暗男一同离开。",
        "摩西和暗兰一同离开。",
    ) is True
    assert _should_accept_context_sentence_correction(
        "摩西和暗男一同离开。",
        "流便扫罗迦南女子暗兰约基别。",
    ) is False


def test_context_exception_trace_fields_never_include_exception_text():
    secret = "SECRET_CONTEXT_TERM"
    fields = _safe_exception_trace_fields(RuntimeError(f"request prompt: {secret}"))

    assert fields["error_type"] == "RuntimeError"
    assert len(fields["error_sha256"]) == 64
    assert secret not in json.dumps(fields)


def test_cancelled_thread_bridge_waits_for_worker_completion():
    async def scenario():
        started = threading.Event()
        release = threading.Event()

        def worker():
            started.set()
            release.wait(timeout=2.0)
            return "finished"

        task = asyncio.create_task(_await_thread_completion_on_cancel(worker))
        assert await asyncio.to_thread(started.wait, 1.0)
        task.cancel()
        await asyncio.sleep(0.05)
        assert task.done() is False
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_should_skip_stream_decode_for_clean_silence_without_pending_text():
    assert _should_skip_stream_decode(
        in_speech=False,
        silence_ms=1200.0,
        segment_elapsed_ms=1200.0,
        snr_db=2.0,
        vad_silence_ms=900.0,
        vad_exit_snr_db=4.0,
        has_pending_text=False,
    )


def test_should_skip_stream_decode_when_pure_silence_has_no_speech_phase_yet():
    assert _should_skip_stream_decode(
        in_speech=False,
        silence_ms=0.0,
        segment_elapsed_ms=1500.0,
        snr_db=1.0,
        vad_silence_ms=900.0,
        vad_exit_snr_db=4.0,
        has_pending_text=False,
    )


def test_should_skip_stream_decode_even_when_pending_text_exists():
    assert _should_skip_stream_decode(
        in_speech=False,
        silence_ms=1500.0,
        segment_elapsed_ms=1500.0,
        snr_db=2.0,
        vad_silence_ms=900.0,
        vad_exit_snr_db=4.0,
        has_pending_text=True,
    )


def test_should_not_skip_stream_decode_when_snr_is_high():
    assert not _should_skip_stream_decode(
        in_speech=False,
        silence_ms=1500.0,
        segment_elapsed_ms=1500.0,
        snr_db=7.5,
        vad_silence_ms=900.0,
        vad_exit_snr_db=4.0,
        has_pending_text=False,
    )


def test_should_skip_stream_decode_for_in_speech_trailing_silence():
    assert _should_skip_stream_decode(
        in_speech=True,
        silence_ms=160.0,
        segment_elapsed_ms=2400.0,
        snr_db=2.5,
        vad_silence_ms=900.0,
        vad_exit_snr_db=4.0,
        has_pending_text=True,
    )


def test_should_not_skip_stream_decode_for_in_speech_tiny_silence():
    assert not _should_skip_stream_decode(
        in_speech=True,
        silence_ms=40.0,
        segment_elapsed_ms=2400.0,
        snr_db=2.5,
        vad_silence_ms=900.0,
        vad_exit_snr_db=4.0,
        has_pending_text=True,
    )


def test_should_use_high_batch_merge_under_backpressure_even_with_shallow_depth():
    assert _should_use_high_batch_merge(queue_depth=12, audio_queue_size=64, under_pressure=True)


def test_should_not_use_high_batch_merge_when_queue_is_light_and_no_pressure():
    assert not _should_use_high_batch_merge(queue_depth=6, audio_queue_size=64, under_pressure=False)


def test_pending_prefix_duplicate_filter_keeps_exact_repeated_sentence():
    # Pending prefix equal to one full sentence can be a real repetition;
    # do not treat it as carry-over duplicate.
    assert not _is_probable_pending_prefix_duplicate(
        "这是什么题目？",
        "这是什么题目？",
        "这是什么",
        "这是什么题目？",
    )


def test_pending_prefix_duplicate_filter_only_on_tiny_prefix_with_extra_tail():
    assert _is_probable_pending_prefix_duplicate(
        "这是什么题目？",
        "这是什么题目？",
        "这",
        "这是什么题目？这又是什么题目？",
    )


def test_carry_overlap_skip_requires_at_least_two_overlapped_sentences():
    assert not _should_apply_carry_overlap_skip(overlap_count=1, overlap_chars=36, raw_chars=8)
    assert _should_apply_carry_overlap_skip(overlap_count=2, overlap_chars=36, raw_chars=8)


def test_carry_overlap_skip_rejects_long_raw_text():
    assert not _should_apply_carry_overlap_skip(overlap_count=2, overlap_chars=36, raw_chars=28)


def test_partial_reset_guard_detects_suspicious_short_rewrite():
    assert _should_hold_partial_reset(
        prev_text="在耶和华神所造的所有活物当中，蛇是最狡猾的。蛇对女人说。",
        next_text="神对女人。",
    )


def test_partial_reset_guard_ignores_normal_growth_text():
    assert not _should_hold_partial_reset(
        prev_text="第一遍测试翻译，第二遍测试翻译",
        next_text="第一遍测试翻译，第二遍测试翻译，第三遍测试翻译",
    )


def test_partial_reset_guard_ignores_short_previous_text():
    assert not _should_hold_partial_reset(
        prev_text="第一遍。",
        next_text="第二遍。",
    )


def test_partial_reset_release_requires_hits_or_timeout():
    assert not _should_release_partial_reset_guard(candidate_hits=1, hold_sec=0.4)
    assert _should_release_partial_reset_guard(candidate_hits=2, hold_sec=0.4)
    assert _should_release_partial_reset_guard(candidate_hits=1, hold_sec=1.3)


def test_short_english_sentence_early_commit_guard_blocks_short_heading_fragment():
    assert _is_short_english_sentence_for_early_commit("The Short Session Topic.")


def test_short_english_sentence_early_commit_guard_allows_long_english_and_cjk():
    assert not _is_short_english_sentence_for_early_commit(
        "A complete longer English sentence contains enough words to be committed safely."
    )
    assert not _is_short_english_sentence_for_early_commit("这是一个已经稳定完成并且长度足够的句子。")


def test_split_sentences_treats_three_letter_name_as_terminal_boundary():
    sentences, tail = _split_sentences_and_tail(
        "This conversation is with me and my husband Dan. You asked a useful question."
    )
    assert sentences == [
        "This conversation is with me and my husband Dan.",
        "You asked a useful question.",
    ]
    assert tail == ""


def test_split_sentences_keeps_initials_and_decimal_periods_internal():
    sentences, tail = _split_sentences_and_tail(
        "The U.S. rate was 3.14 percent. The report ended."
    )
    assert sentences == ["The U.S. rate was 3.14 percent.", "The report ended."]
    assert tail == ""


def test_boundary_join_mode_distinguishes_overlap_spacing_and_direct_join():
    classifier = getattr(demo_streaming_ws, "_classify_boundary_join_mode", None)
    assert classifier is not None
    assert classifier("repeat text", "text continues", "repeat text continues") == "overlap"
    assert classifier("She's a girl", "Yes.", "She's a girl Yes.") == "spaced"
    assert classifier("上半句", "下半句", "上半句下半句") == "direct"


def test_short_english_slice_fragment_guard_blocks_period_fragments_only():
    guard = demo_streaming_ws._is_short_english_slice_fragment
    assert guard("Short fragment.")
    assert guard("The Short Session Topic.")
    assert not guard("A complete longer English sentence contains enough words.")
    assert not guard("Are you ready?")
    assert not guard("这是一个已经稳定完成并且长度足够的句子。")


def test_hard_cut_fallback_does_not_merge_completed_prefix_with_unrelated_raw():
    assert hasattr(demo_streaming_ws, "_should_hard_cut_fallback_merge")
    assert not demo_streaming_ws._should_hard_cut_fallback_merge(
        'A completed quoted sentence."',
        "A different unrelated sentence should stay separate.",
    )


def test_hard_cut_fallback_merges_unfinished_cjk_tail():
    assert hasattr(demo_streaming_ws, "_should_hard_cut_fallback_merge")
    assert demo_streaming_ws._should_hard_cut_fallback_merge("第一句不完整", "继续补全成句。")


def test_openai_api_translator_retries_when_generation_hits_token_limit(monkeypatch):
    calls = []

    class _FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def fake_urlopen(req, timeout):
        body = json.loads(req.data.decode("utf-8"))
        calls.append(body)
        if len(calls) == 1:
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "截断输出"},
                        }
                    ]
                }
            )
        return _FakeResponse(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "完整输出"},
                    }
                ]
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    translator = OpenAIAPITranslator(
        "http://127.0.0.1:8001",
        "fake-model",
        source_language="English",
        target_language="中文",
        max_new_tokens=64,
    )

    assert translator.translate("A long English sentence.", "English", "中文") == "完整输出"
    assert [call["max_tokens"] for call in calls] == [64, 128]


def test_find_first_boundary_after_prefers_latest_boundary():
    pattern = re.compile(r"[.!?]")
    found = _find_first_boundary_after("a.b.c.", 0, pattern)
    assert found == (6, ".")


def test_find_first_boundary_after_respects_start_offset():
    pattern = re.compile(r"[.!?]")
    found = _find_first_boundary_after("a.b.c.", 3, pattern)
    assert found == (6, ".")


def test_alignment_summary_reports_model_seen_not_committed():
    model_seen = {}
    committed_seen = {}
    _alignment_registry_touch(model_seen, "第一句。", 10, "partial_raw")
    _alignment_registry_touch(model_seen, "第一句。", 11, "partial_raw")
    _alignment_registry_touch(model_seen, "第二句。", 12, "partial_raw")
    _alignment_registry_touch(model_seen, "第二句。", 13, "final_raw")
    _alignment_registry_touch(committed_seen, "第一句。", 20, "sentence_committed")

    summary = _summarize_alignment_gap(model_seen, committed_seen, min_model_hits=2, max_samples=4)
    assert summary["model_all_unique"] == 2
    assert summary["model_stable_unique"] == 2
    assert summary["model_final_unique"] == 1
    assert summary["committed_unique"] == 1
    assert summary["missing_unique"] == 1
    assert summary["missing_samples"][0]["text"] == "第二句。"
    assert summary["final_missing_unique"] == 1
    assert summary["final_missing_samples"][0]["text"] == "第二句。"


def test_alignment_summary_treats_final_raw_single_hit_as_stable():
    model_seen = {}
    committed_seen = {}
    _alignment_registry_touch(model_seen, "最终句子。", 30, "final_raw")

    summary = _summarize_alignment_gap(model_seen, committed_seen, min_model_hits=2, max_samples=4)
    assert summary["model_stable_unique"] == 1
    assert summary["model_final_unique"] == 1
    assert summary["missing_unique"] == 1
    assert summary["final_missing_unique"] == 1


def test_alignment_summary_final_missing_only_counts_final_raw_sentences():
    model_seen = {}
    committed_seen = {}
    _alignment_registry_touch(model_seen, "仅partial句子。", 10, "partial_raw")
    _alignment_registry_touch(model_seen, "仅partial句子。", 11, "partial_raw")
    summary = _summarize_alignment_gap(model_seen, committed_seen, min_model_hits=2, max_samples=4)
    assert summary["missing_unique"] == 1
    assert summary["final_missing_unique"] == 0
    assert summary["final_missing_samples"] == []


def test_alignment_touch_model_collapses_incremental_growth():
    model_seen = {}
    _alignment_registry_touch_model(model_seen, "第四遍测试翻译，第五遍。", 10, "partial_raw")
    _alignment_registry_touch_model(model_seen, "第四遍测试翻译，第五遍测试翻译。", 11, "partial_raw")
    assert len(model_seen) == 1
    only = next(iter(model_seen.values()))
    assert "第五遍测试翻译" in str(only.get("text", ""))
    assert int(only.get("hits", 0) or 0) >= 2


def test_alignment_touch_model_attaches_short_regression_to_longer_sentence():
    model_seen = {}
    _alignment_registry_touch_model(model_seen, "第四遍测试翻译，第五遍测试翻译。", 10, "partial_raw")
    _alignment_registry_touch_model(model_seen, "第四遍测试翻译，第五遍。", 11, "partial_raw")
    assert len(model_seen) == 1
    only = next(iter(model_seen.values()))
    assert "第五遍测试翻译" in str(only.get("text", ""))
    assert int(only.get("hits", 0) or 0) >= 2


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


def test_parse_args_uses_safe_asr_context_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])

    args = parse_args()

    assert args.asr_context_schedule == ""
    assert args.asr_context_max_terms == 24
    assert args.asr_context_max_chars == 160
    assert args.asr_context_lookaround_sec == 30.0
    assert args.asr_context_apply_mode == "segment_final"


def test_parse_args_accepts_asr_context_options(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--asr-context-schedule",
            "/tmp/context.json",
            "--asr-context-max-terms",
            "12",
            "--asr-context-max-chars",
            "80",
            "--asr-context-lookaround-sec",
            "15",
            "--asr-context-apply-mode",
            "streaming",
        ],
    )

    args = parse_args()

    assert args.asr_context_schedule == "/tmp/context.json"
    assert args.asr_context_max_terms == 12
    assert args.asr_context_max_chars == 80
    assert args.asr_context_lookaround_sec == 15.0
    assert args.asr_context_apply_mode == "streaming"


def test_parse_args_accepts_audio_queue_size(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--audio-queue-size", "12"],
    )
    args = parse_args()
    assert args.audio_queue_size == 12


def test_parse_args_uses_balanced_early_translation_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])
    args = parse_args()
    assert args.translation_max_new_tokens == 128
    assert args.early_translation_stable_sec == 0.8
    assert args.early_translation_stable_hits == 3
    assert args.early_translation_short_stable_sec == 1.2
    assert args.early_translation_short_stable_hits == 4


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
            "--backpressure-hard-relief-sec",
            "7.2",
        ],
    )
    args = parse_args()
    assert args.segment_hard_cut_sec == 28.0
    assert args.segment_overlap_sec == 0.9
    assert args.backpressure_target_queue_sec == 3.2
    assert args.backpressure_max_queue_sec == 5.7
    assert args.backpressure_hard_relief_sec == 7.2


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


def test_auth_password_hash_verifies_and_rejects_wrong_password():
    encoded = _hash_auth_password("secret")
    assert encoded.startswith("pbkdf2_sha256$")
    assert _verify_auth_password("secret", encoded)
    assert not _verify_auth_password("wrong", encoded)
    assert not _verify_auth_password("secret", "not-a-valid-hash")


def test_parse_args_accepts_auth_options(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--auth-enabled",
            "--auth-username",
            "operator",
            "--auth-password-hash",
            "pbkdf2_sha256$1$c2FsdA$ZGlnZXN0",
            "--auth-cookie-secure",
            "--auth-session-ttl-sec",
            "7200",
            "--disable-debug-file",
        ],
    )
    args = parse_args()
    assert args.auth_enabled is True
    assert args.auth_username == "operator"
    assert args.auth_password_hash == "pbkdf2_sha256$1$c2FsdA$ZGlnZXN0"
    assert args.auth_cookie_secure is True
    assert args.auth_session_ttl_sec == 7200
    assert args.disable_debug_file is True


def test_parse_args_accepts_early_translation_stability_controls(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--early-translation-stable-sec",
            "0.4",
            "--early-translation-stable-hits",
            "3",
            "--early-translation-short-stable-sec",
            "0.9",
            "--early-translation-short-stable-hits",
            "5",
            "--early-translation-min-english-words",
            "7",
            "--early-translation-min-english-chars",
            "36",
        ],
    )
    args = parse_args()
    assert args.early_translation_stable_sec == 0.4
    assert args.early_translation_stable_hits == 3
    assert args.early_translation_short_stable_sec == 0.9
    assert args.early_translation_short_stable_hits == 5
    assert args.early_translation_min_english_words == 7
    assert args.early_translation_min_english_chars == 36


def test_index_template_contains_core_stream_controls():
    assert 'id="btnStart"' in INDEX_HTML_TEMPLATE
    assert 'id="btnStop"' in INDEX_HTML_TEMPLATE
    assert 'id="status"' in INDEX_HTML_TEMPLATE
    assert 'id="text"' in INDEX_HTML_TEMPLATE
    assert 'id="translation"' in INDEX_HTML_TEMPLATE


def test_index_template_uses_eye_friendly_light_theme():
    assert "--bg-a:#edf4ea" in INDEX_HTML_TEMPLATE
    assert "--bg-b:#f6f0e6" in INDEX_HTML_TEMPLATE
    assert "--surface:#f8fbf4" in INDEX_HTML_TEMPLATE
    assert "#0f1114" not in INDEX_HTML_TEMPLATE
    assert "linear-gradient(180deg, #2b3139" not in INDEX_HTML_TEMPLATE


def test_index_template_auto_hides_top_controls_while_running():
    assert 'id="appCard"' in INDEX_HTML_TEMPLATE
    assert 'id="controlBar"' in INDEX_HTML_TEMPLATE
    assert 'id="controlReveal"' in INDEX_HTML_TEMPLATE
    assert "function setControlBarHidden" in INDEX_HTML_TEMPLATE
    assert "function revealControlBarTemporarily" in INDEX_HTML_TEMPLATE
    assert 'setControlBarHidden(true, "start_success");' in INDEX_HTML_TEMPLATE
    assert 'setControlBarHidden(false, "final");' in INDEX_HTML_TEMPLATE
    assert 'setControlBarHidden(false, "start_failed");' in INDEX_HTML_TEMPLATE


def test_index_template_has_configurable_subtitle_font_controls():
    assert 'id="subtitleTopFontInput"' in INDEX_HTML_TEMPLATE
    assert 'id="subtitleBottomFontInput"' in INDEX_HTML_TEMPLATE
    assert "--subtitle-top-font-size" in INDEX_HTML_TEMPLATE
    assert "--subtitle-bottom-font-size" in INDEX_HTML_TEMPLATE
    assert 'font-size: var(--subtitle-top-font-size' in INDEX_HTML_TEMPLATE
    assert 'font-size: var(--subtitle-bottom-font-size' in INDEX_HTML_TEMPLATE
    assert "const SUBTITLE_TOP_FONT_KEY" in INDEX_HTML_TEMPLATE
    assert "const SUBTITLE_BOTTOM_FONT_KEY" in INDEX_HTML_TEMPLATE
    assert "function applySubtitleFontSizes" in INDEX_HTML_TEMPLATE
    assert "function readSubtitleFontConfig" in INDEX_HTML_TEMPLATE
    assert "subtitleTopFontInput.addEventListener" in INDEX_HTML_TEMPLATE
    assert "subtitleBottomFontInput.addEventListener" in INDEX_HTML_TEMPLATE


def test_index_template_keeps_mobile_latest_subtitles_visible():
    assert "height: 100svh;" in INDEX_HTML_TEMPLATE
    assert "@supports (height: 100dvh)" in INDEX_HTML_TEMPLATE
    assert ".card.controls-hidden" in INDEX_HTML_TEMPLATE
    assert "grid-template-rows: auto minmax(0, 1fr);" in INDEX_HTML_TEMPLATE
    assert "@media (max-width: 720px)" in INDEX_HTML_TEMPLATE


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
    assert "const committedText = String(msg.committed_text || \"\").trim();" in INDEX_HTML_TEMPLATE
    assert "const tentativeTail = resolveTentativeTail(" in INDEX_HTML_TEMPLATE
    assert "if (USE_COMMITTED_SENTENCE_EVENTS) {" in INDEX_HTML_TEMPLATE
    assert "const rows = committedRows.slice();" in INDEX_HTML_TEMPLATE
    assert 'rows.push({ sid: "__tail__", zh: tail, en: "" });' in INDEX_HTML_TEMPLATE


def test_index_template_keeps_existing_lane_density_while_allowing_scroll():
    assert "const MAX_VISIBLE_ROWS_ZH = 4;" in INDEX_HTML_TEMPLATE
    assert "const MAX_VISIBLE_ROWS_EN = MAX_VISIBLE_ROWS_ZH + 2;" in INDEX_HTML_TEMPLATE
    assert "overflow-y: auto;" in INDEX_HTML_TEMPLATE


def test_index_template_keeps_full_history_rows_in_committed_mode():
    assert "function clipVisibleRows" not in INDEX_HTML_TEMPLATE
    assert "return rows;" in INDEX_HTML_TEMPLATE


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


def test_index_template_uses_backend_stability_for_tentative_tail():
    assert "const TAIL_STABILIZE_MS = 700;" not in INDEX_HTML_TEMPLATE
    assert "tailStabilizeTimer = setTimeout" not in INDEX_HTML_TEMPLATE
    assert "function readBackendStability" in INDEX_HTML_TEMPLATE
    assert "function updateCommittedTentativeTailFromBackend" in INDEX_HTML_TEMPLATE
    assert "readBackendStability(msg)" in INDEX_HTML_TEMPLATE
    assert "source.stability" in INDEX_HTML_TEMPLATE
    assert 'rows.push({ sid: "__tail__", zh: tail, en: "" });' in INDEX_HTML_TEMPLATE


def test_index_template_uses_stop_only_finish_mode():
    assert "const STOP_FINAL_TIMEOUT_MS = 120000;" in INDEX_HTML_TEMPLATE
    assert "const payload = {type: \"finish\", mode};" in INDEX_HTML_TEMPLATE
    assert "if (reason) payload.reason = String(reason);" in INDEX_HTML_TEMPLATE
    assert "ws.send(JSON.stringify(payload));" in INDEX_HTML_TEMPLATE
    assert 'await sendFinishAndAwaitFinal("stop", STOP_FINAL_TIMEOUT_MS);' in INDEX_HTML_TEMPLATE
    assert 'sock.send(JSON.stringify({type: "finish", mode: "stop"}));' in INDEX_HTML_TEMPLATE


def test_index_template_blocks_start_reentry_while_awaiting_final():
    assert "if (running || awaitingFinal) return;" in INDEX_HTML_TEMPLATE
    assert "awaitingFinal = true;" in INDEX_HTML_TEMPLATE
    assert 'setStatus("Finishing / 收尾中", "warn");' in INDEX_HTML_TEMPLATE
    assert 'setStatus("Stopped / 已停止", "");' in INDEX_HTML_TEMPLATE
    assert "lockUI(false);" in INDEX_HTML_TEMPLATE


def test_index_template_keeps_finishing_on_stop_final_timeout():
    assert "if (msg.includes(\"final timeout\")) {" in INDEX_HTML_TEMPLATE
    assert 'setStatus("Finishing (slow backend) / 收尾中(后端较慢)", "warn");' in INDEX_HTML_TEMPLATE
    assert "return;" in INDEX_HTML_TEMPLATE


def test_index_template_single_stream_state_no_frontend_slice_reopen():
    assert "async function rotateSliceSession(reason = \"time\"){" not in INDEX_HTML_TEMPLATE
    assert "await openSocket();" in INDEX_HTML_TEMPLATE
    assert 'type: "start"' in INDEX_HTML_TEMPLATE
    assert "language: selectedAsrLanguage()" in INDEX_HTML_TEMPLATE
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


def test_backend_finish_preserves_audio_queue_for_tail_accuracy():
    src = Path(__file__).resolve().parents[1] / "voxbridge" / "cli" / "demo_streaming_ws.py"
    text = src.read_text(encoding="utf-8")
    assert "ws finish drops pending queue" not in text
    assert "_trace_event(\"finish_queue_cleared\"" not in text
    assert "finish_queue_preserved" in text


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
    assert "updateCommittedTentativeTailFromBackend(tentativeTail, stability);" in INDEX_HTML_TEMPLATE
    assert "function resolveTentativeTail(nextText, committedText, tentativeText){" in INDEX_HTML_TEMPLATE
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


def test_split_sentences_uses_generic_closing_quote_boundaries():
    text = 'Alpha beta." Gamma delta? Zeta eta!'
    sentences, tail = _split_sentences_and_tail(text)
    assert sentences == ['Alpha beta."', "Gamma delta?", "Zeta eta!"]
    assert tail == ""


def test_split_sentences_keeps_generic_initialism_and_decimal_together():
    text = "The U.S. marker is version 3.14 today. Next sentence."
    sentences, tail = _split_sentences_and_tail(text)
    assert sentences == ["The U.S. marker is version 3.14 today.", "Next sentence."]
    assert tail == ""


def test_split_sentences_handles_mixed_cjk_latin_strong_boundaries():
    text = '这是第一句已经足够长。Alpha beta."这是第三句已经足够长！'
    sentences, tail = _split_sentences_and_tail(text)
    assert sentences == ["这是第一句已经足够长。", 'Alpha beta."', "这是第三句已经足够长！"]
    assert tail == ""


def test_trim_leading_boundary_overlap_removes_cjk_suffix_replay():
    assert (
        _trim_leading_boundary_overlap(
            "已经有火从天上降下来，烧灭前两次来的五十人。",
            "人。现在，愿我的性命在你眼前看为宝贵。",
        )
        == "现在，愿我的性命在你眼前看为宝贵。"
    )
    assert (
        _trim_leading_boundary_overlap(
            "过去之后，他说：“你要我为你做什么？”",
            "什么？只管求我。",
        )
        == "只管求我。"
    )


def test_trim_leading_boundary_overlap_requires_terminal_replay():
    assert _trim_leading_boundary_overlap("这是上一句的尾字人。", "人现在继续说。") == "人现在继续说。"
    assert _trim_leading_boundary_overlap("这是上一句。", "这是下一句。") == "这是下一句。"


def test_should_accept_sentence_upgrade_allows_growth():
    assert _should_accept_sentence_upgrade("第一遍测试翻译。", "第一遍测试翻译，第二遍测试翻译。")


def test_should_accept_sentence_upgrade_allows_corrected_tail_growth():
    old = (
        "But I would have ultimately been okay with it because yeah, like when you take out "
        "a piece of paper and you start writing down, why do I want to."
    )
    new = (
        "But I would have ultimately been okay with it because yeah, like when you take out "
        "a piece of paper and you start writing down, why do I want a third kid, and you're "
        "like, I don't even know, this is going to be a lot more work, and we're going to "
        "have to have a minivan forever and stuff like that."
    )
    assert _should_accept_sentence_upgrade(old, new)


def test_should_accept_sentence_upgrade_rejects_middle_rewrite_with_shared_intro():
    old = "This is a carefully completed sentence about pregnancy planning."
    new = (
        "This is a carefully completed sentence about retirement accounts, investment risk, "
        "and an unrelated subject that continues for much longer."
    )
    assert not _should_accept_sentence_upgrade(old, new)


def test_should_accept_sentence_upgrade_allows_quality_fix_without_growth():
    assert _should_accept_sentence_upgrade("第五。次测试翻译。", "第五次测试翻译。")


def test_should_accept_sentence_upgrade_rejects_lower_quality_variant():
    assert not _should_accept_sentence_upgrade("第五次测试翻译。", "第五。次测试翻译。")


def test_should_accept_sentence_upgrade_rejects_unrelated_shorter_text():
    assert not _should_accept_sentence_upgrade("这是一段完整的测试文本。", "这是测试。")


def test_monotonic_sentence_extension_accepts_short_terminal_suffixes():
    classifier = getattr(demo_streaming_ws, "_is_monotonic_sentence_extension", None)
    assert classifier is not None
    assert classifier(
        "The result is ready.",
        "The result is ready now.",
    )
    assert classifier("结果已经确认。", "结果已经确认完成。")


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("The result is ready.", "The result is ready."),
        ("The result is ready now.", "The result is ready."),
        ("The report is ready.", "The report was rejected today."),
        ("Fifth test completed.", "Fifth. test completed."),
    ],
)
def test_monotonic_sentence_extension_rejects_non_growth_and_rewrites(old, new):
    classifier = getattr(demo_streaming_ws, "_is_monotonic_sentence_extension", None)
    assert classifier is not None
    assert not classifier(old, new)


def test_deferred_sentence_upgrade_requires_hits_and_elapsed_stability():
    observer = getattr(demo_streaming_ws, "_observe_deferred_sentence_upgrade", None)
    assert observer is not None
    candidates = {}

    first = observer(
        candidates,
        sentence_id="sentence-1",
        text="The result is ready now.",
        seq=10,
        now=100.0,
        required_hits=3,
        required_stable_sec=0.6,
    )
    second = observer(
        candidates,
        sentence_id="sentence-1",
        text="The result is ready now.",
        seq=11,
        now=100.3,
        required_hits=3,
        required_stable_sec=0.6,
    )
    third = observer(
        candidates,
        sentence_id="sentence-1",
        text="The result is ready now.",
        seq=12,
        now=100.61,
        required_hits=3,
        required_stable_sec=0.6,
    )

    assert (first.transition, first.ready, first.hits) == ("started", False, 1)
    assert (second.transition, second.ready, second.hits) == ("waiting", False, 2)
    assert (third.transition, third.ready, third.hits) == ("accepted", True, 3)
    assert third.stable_ms == 610


def test_deferred_sentence_upgrade_change_restarts_stability_window():
    observer = getattr(demo_streaming_ws, "_observe_deferred_sentence_upgrade", None)
    assert observer is not None
    candidates = {}
    observer(
        candidates,
        "sentence-1",
        "The result is ready so.",
        10,
        100.0,
        3,
        0.6,
    )
    changed = observer(
        candidates,
        "sentence-1",
        "The result is ready now.",
        11,
        100.7,
        3,
        0.6,
    )

    assert changed.transition == "changed"
    assert changed.ready is False
    assert changed.hits == 1
    assert changed.previous_text == "The result is ready so."


def test_index_template_hides_raw_asr_text_panel():
    assert "Raw ASR Text" not in INDEX_HTML_TEMPLATE
    assert 'id="rawText"' not in INDEX_HTML_TEMPLATE
    assert "class=\"raw-panel\"" not in INDEX_HTML_TEMPLATE


def test_index_template_has_translation_direction_selector_and_ws_control():
    assert 'id="translationDirectionSelect"' in INDEX_HTML_TEMPLATE
    assert 'id="translationDirectionLabel"' in INDEX_HTML_TEMPLATE
    assert 'value="zh2en"' in INDEX_HTML_TEMPLATE
    assert 'value="en2zh"' in INDEX_HTML_TEMPLATE
    assert 'type="checkbox"' not in INDEX_HTML_TEMPLATE
    assert "function selectedTranslationDirection()" in INDEX_HTML_TEMPLATE
    assert 'type: "set_translation_direction"' in INDEX_HTML_TEMPLATE
    assert "translation_direction: selectedTranslationDirection()" in INDEX_HTML_TEMPLATE


def test_index_template_uses_translation_source_language_for_asr_start():
    assert "function selectedAsrLanguage()" in INDEX_HTML_TEMPLATE
    assert 'return selectedTranslationDirection() === "en2zh" ? "English" : "Chinese";' in INDEX_HTML_TEMPLATE
    assert "language: selectedAsrLanguage()" in INDEX_HTML_TEMPLATE
    assert "language: selectedLanguage()" not in INDEX_HTML_TEMPLATE


def test_index_template_locks_translation_direction_during_active_session():
    assert "if (translationDirectionSelect) translationDirectionSelect.disabled = active;" in INDEX_HTML_TEMPLATE
    assert "if (translationDirectionSelect) translationDirectionSelect.disabled = true;" in INDEX_HTML_TEMPLATE


def test_index_template_has_audio_input_source_selector():
    assert 'id="inputSourceSelect"' in INDEX_HTML_TEMPLATE
    assert 'id="inputSourceLabel"' in INDEX_HTML_TEMPLATE
    assert 'value="mic"' in INDEX_HTML_TEMPLATE
    assert 'value="system"' in INDEX_HTML_TEMPLATE
    assert "function selectedInputSource()" in INDEX_HTML_TEMPLATE


def test_index_template_supports_scrollable_subtitle_history():
    assert "MAX_SUBTITLE_HISTORY = 100" in INDEX_HTML_TEMPLATE
    assert "trimSubtitleHistory(" in INDEX_HTML_TEMPLATE
    assert "history_trimmed" in INDEX_HTML_TEMPLATE


def test_index_template_has_lane_auto_follow_state_machine():
    assert "scrollFollowState" in INDEX_HTML_TEMPLATE
    assert "bindSubtitleScrollTracking(" in INDEX_HTML_TEMPLATE
    assert "pauseSubtitleAutoFollow(" in INDEX_HTML_TEMPLATE
    assert "resumeSubtitleAutoFollow(" in INDEX_HTML_TEMPLATE
    assert "scroll_follow_paused" in INDEX_HTML_TEMPLATE
    assert "scroll_follow_resumed" in INDEX_HTML_TEMPLATE


def test_index_template_has_scroll_to_latest_buttons_per_lane():
    assert 'id="jumpLatestEn"' in INDEX_HTML_TEMPLATE
    assert 'id="jumpLatestZh"' in INDEX_HTML_TEMPLATE
    assert "function updateJumpLatestButtons()" in INDEX_HTML_TEMPLATE
    assert "jump_latest_clicked" in INDEX_HTML_TEMPLATE


def test_index_template_has_persisted_session_context_control():
    assert 'id="asrContextInput"' in INDEX_HTML_TEMPLATE
    assert 'for="asrContextInput"' in INDEX_HTML_TEMPLATE
    assert "专业术语 Context" in INDEX_HTML_TEMPLATE
    assert 'const ASR_CONTEXT_STORAGE_KEY = "voxbridge_asr_context_terms";' in INDEX_HTML_TEMPLATE
    assert "function parseAsrContextTerms" in INDEX_HTML_TEMPLATE
    assert "function readAsrContextTerms" in INDEX_HTML_TEMPLATE
    assert "localStorage.setItem(ASR_CONTEXT_STORAGE_KEY" in INDEX_HTML_TEMPLATE
    assert "localStorage.removeItem(ASR_CONTEXT_STORAGE_KEY)" in INDEX_HTML_TEMPLATE


def test_index_template_sends_context_in_microphone_and_replay_starts():
    assert INDEX_HTML_TEMPLATE.count("asr_context_terms: asrContextTerms") == 2
    assert "const asrContextTerms = readAsrContextTerms();" in INDEX_HTML_TEMPLATE


def test_index_template_locks_context_during_start_active_and_finishing_states():
    assert "if (asrContextInput) asrContextInput.disabled = active;" in INDEX_HTML_TEMPLATE
    assert "if (asrContextInput) asrContextInput.disabled = true;" in INDEX_HTML_TEMPLATE
    assert 'setControlBarHidden(false, "start_failed");' in INDEX_HTML_TEMPLATE


def test_index_template_waits_for_started_before_capture_becomes_active():
    assert "function waitForStarted" in INDEX_HTML_TEMPLATE
    assert "resolvePendingStart(msg);" in INDEX_HTML_TEMPLATE
    assert "rejectPendingStart(new Error(msg.message" in INDEX_HTML_TEMPLATE
    assert "const startedPromise = waitForStarted(10000);" in INDEX_HTML_TEMPLATE
    assert "const started = await startedPromise;" in INDEX_HTML_TEMPLATE
    assert "await buildCaptureGraph();" in INDEX_HTML_TEMPLATE
    assert INDEX_HTML_TEMPLATE.index(
        "const started = await startedPromise;"
    ) < INDEX_HTML_TEMPLATE.index("await buildCaptureGraph();")


def test_index_template_uses_backend_context_limits_and_safe_status_metadata():
    assert "const ASR_CONTEXT_MAX_TERMS = __ASR_CONTEXT_MAX_TERMS__;" in INDEX_HTML_TEMPLATE
    assert "const ASR_CONTEXT_MAX_CHARS = __ASR_CONTEXT_MAX_CHARS__;" in INDEX_HTML_TEMPLATE
    assert "Context 已启用" in INDEX_HTML_TEMPLATE
    assert "function asrContextTermHasSentencePunctuation" in INDEX_HTML_TEMPLATE
    assert "asr_context_term_count" in INDEX_HTML_TEMPLATE
    assert "asr_context_chars" in INDEX_HTML_TEMPLATE
    assert "contextTerms:" not in INDEX_HTML_TEMPLATE


def test_index_template_context_control_uses_own_mobile_row_without_frontend_text_splitting():
    assert ".context-control" in INDEX_HTML_TEMPLATE
    assert "flex: 1 1 100%;" in INDEX_HTML_TEMPLATE
    assert "splitTextByDisplayRules" not in INDEX_HTML_TEMPLATE
    assert "VAD_SILENCE_SEC" not in INDEX_HTML_TEMPLATE


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
