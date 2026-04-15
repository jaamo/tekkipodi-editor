from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from .assemble import mix_tracks
from .dedup import (
    apply_cuts,
    find_retakes,
    parse_review_markdown,
    write_decisions,
    write_review_markdown,
)
from .io_utils import discover_session, require_jingles
from .master import master_audio
from .models import CutSpan
from .silence import strip_silence_with_log
from .transcribe import transcribe_session

_console = Console()

# Opening overlap: intro plays at full volume, ducks to INTRO_DUCKED_GAIN_DB
# linearly over [INTRO_DUCK_START_MS, INTRO_DUCK_END_MS], and the first topic
# starts at INTRO_TOPIC_START_MS on top of the (now-ducked) intro. After
# sitting under the topic at the ducked level, the intro then fades out to
# silence over [INTRO_FADEOUT_START_MS, INTRO_FADEOUT_END_MS]; anything past
# INTRO_FADEOUT_END_MS is truncated so the intro leaves the mix cleanly.
INTRO_TOPIC_START_MS = 5800
INTRO_DUCK_START_MS = 5000
INTRO_DUCK_END_MS = 7000
INTRO_DUCKED_GAIN_DB = -10.0
INTRO_FADEOUT_START_MS = 10000
INTRO_FADEOUT_END_MS = 15000

# Mid-jingle cross-play: the mid jingle starts MID_PREV_OVERLAP_MS before the
# previous segment ends, and the next segment starts MID_NEXT_DELAY_MS after
# the mid jingle started. The gap between consecutive segments is therefore
# (MID_NEXT_DELAY_MS - MID_PREV_OVERLAP_MS) of jingle-solo playback, after
# MID_PREV_OVERLAP_MS of jingle-over-previous-segment overlap.
MID_PREV_OVERLAP_MS = 1400
MID_NEXT_DELAY_MS = 4000

# Target integrated loudness for both jingles *before* they hit the timeline.
# We loudness-normalize each jingle (via pyloudnorm) to this value rather than
# applying a relative gain, so the final jingle level in the mastered episode
# is independent of how loud the source jingle file happens to be. The master
# stage then gain-matches the whole mix to TARGET_LUFS (-16 LUFS in master.py)
# based mostly on the speech, so a jingle normalized to JINGLE_TARGET_LUFS
# ends up at roughly (JINGLE_TARGET_LUFS + (master.TARGET_LUFS - speech_lufs))
# in the final render — with speech typically around -20 LUFS at source, a
# jingle target of -30 LUFS lands the jingle ~10 dB below the speech in the
# mastered output. Tune this single constant to move all jingles louder or
# quieter uniformly. The intro duck (INTRO_DUCKED_GAIN_DB) is still layered
# on top of this, so the ducked intro tail is ~10 dB below JINGLE_TARGET_LUFS.
JINGLE_TARGET_LUFS = -30.0


def _normalize_loudness(
    audio: "AudioSegment", target_lufs: float
) -> "AudioSegment":
    """Apply a linear gain so that `audio`'s integrated loudness (measured
    via pyloudnorm, the same meter `master.py` uses for the episode-wide
    pass) matches `target_lufs`. If the measurement is not finite (silent
    or too short for the meter), the audio is returned unchanged."""
    import math

    import numpy as np
    import pyloudnorm as pyln

    samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels))
    # pydub samples are signed integers in [-2**(bits-1), 2**(bits-1)-1];
    # pyloudnorm expects floats in roughly [-1.0, 1.0].
    scale = float(1 << (8 * audio.sample_width - 1))
    samples /= scale

    meter = pyln.Meter(audio.frame_rate)
    loudness = meter.integrated_loudness(samples)
    if not math.isfinite(loudness):
        return audio
    return audio + (target_lufs - loudness)


def _duck_intro(intro: "AudioSegment") -> "AudioSegment":
    """Apply the intro duck profile and the subsequent fade-out to silence:
    full volume up to INTRO_DUCK_START_MS, linear duck to INTRO_DUCKED_GAIN_DB
    across [INTRO_DUCK_START_MS, INTRO_DUCK_END_MS], hold, then linearly fade
    to silence over [INTRO_FADEOUT_START_MS, INTRO_FADEOUT_END_MS]. The intro
    is truncated at INTRO_FADEOUT_END_MS so its tail doesn't linger."""
    if len(intro) > INTRO_DUCK_END_MS:
        head = intro[:INTRO_DUCK_START_MS]
        fade_len = INTRO_DUCK_END_MS - INTRO_DUCK_START_MS
        fade = intro[INTRO_DUCK_START_MS:INTRO_DUCK_END_MS].fade(
            from_gain=0.0, to_gain=INTRO_DUCKED_GAIN_DB, start=0, end=fade_len
        )
        tail = intro[INTRO_DUCK_END_MS:] + INTRO_DUCKED_GAIN_DB
        ducked = head + fade + tail
    else:
        ducked = intro

    if len(ducked) > INTRO_FADEOUT_END_MS:
        ducked = ducked[:INTRO_FADEOUT_END_MS]
    tail_after_start = len(ducked) - INTRO_FADEOUT_START_MS
    if tail_after_start > 0:
        fade_dur = min(
            INTRO_FADEOUT_END_MS - INTRO_FADEOUT_START_MS, tail_after_start
        )
        ducked = ducked.fade_out(duration=fade_dur)
    return ducked


def _build_episode_tracks(
    intro_path: Path, mid_path: Path, segment_paths: list[Path]
) -> list[tuple["AudioSegment", int]]:
    """Build the (audio, absolute_position_ms) track list for the whole
    episode: ducked intro at t=0, first topic overlaid at INTRO_TOPIC_START_MS,
    and each subsequent segment separated from the previous one by a mid
    jingle cross-play (see MID_* constants)."""
    from pydub import AudioSegment

    if not segment_paths:
        raise ValueError("Cannot build an episode with zero segments")

    segments = [AudioSegment.from_file(p) for p in segment_paths]
    fr, ch = segments[0].frame_rate, segments[0].channels

    def coerce(a: AudioSegment) -> AudioSegment:
        return a.set_frame_rate(fr).set_channels(ch)

    intro = _normalize_loudness(
        coerce(AudioSegment.from_file(intro_path)), JINGLE_TARGET_LUFS
    )
    mid = _normalize_loudness(
        coerce(AudioSegment.from_file(mid_path)), JINGLE_TARGET_LUFS
    )
    segments = [coerce(s) for s in segments]

    tracks: list[tuple[AudioSegment, int]] = [
        (_duck_intro(intro), 0),
        (segments[0], INTRO_TOPIC_START_MS),
    ]
    prev_end = INTRO_TOPIC_START_MS + len(segments[0])

    for seg in segments[1:]:
        mid_pos = prev_end - MID_PREV_OVERLAP_MS
        next_pos = mid_pos + MID_NEXT_DELAY_MS
        tracks.append((mid, mid_pos))
        tracks.append((seg, next_pos))
        prev_end = next_pos + len(seg)

    return tracks


def process_session(
    session_root: Path,
    output: Path | None = None,
    model_size: str = "small",
    dedup_threshold: float = 0.9,
    dedup_enabled: bool = True,
) -> Path:
    session = discover_session(session_root)
    if output is None:
        output = session.output_dir / "final.wav"
    _console.rule(f"[bold]podtool[/bold] {session_root.name}")
    _console.log(f"{len(session.segments)} segments found")

    transcripts = transcribe_session(session, model_size=model_size)

    auto_spans: list[CutSpan] = []
    if dedup_enabled:
        for segment in session.segments:
            auto_spans.extend(
                find_retakes(
                    segment.stem,
                    transcripts[segment.stem],
                    threshold=dedup_threshold,
                )
            )

    write_review_markdown(session, transcripts, auto_spans)
    _console.log(
        f"wrote review file → {session.review_path.relative_to(session.root)}"
    )
    try:
        input(
            f"\nEdit {session.review_path} to adjust cuts, "
            "then press Enter to continue… "
        )
    except (EOFError, KeyboardInterrupt):
        _console.log("[yellow]aborted at review step[/yellow]")
        raise

    all_spans = parse_review_markdown(session, transcripts)
    _console.log(
        f"review: {len(all_spans)} cut span(s) "
        f"(auto flagged {len(auto_spans)})"
    )

    spans_by_segment: dict[str, list[CutSpan]] = {}
    for span in all_spans:
        spans_by_segment.setdefault(span.segment, []).append(span)

    silence_log: list[dict] = []
    intermediate_paths: list[Path] = []
    for segment in session.segments:
        spans = spans_by_segment.get(segment.stem, [])

        from pydub import AudioSegment

        if spans:
            audio = apply_cuts(segment.path, spans)
            _console.log(f"dedup {segment.stem}: removed {len(spans)} span(s)")
        else:
            audio = AudioSegment.from_file(segment.path)

        pre_len_ms = len(audio)
        trimmed, silence_cuts = strip_silence_with_log(audio)
        intermediate_path = session.intermediate_dir / f"{segment.stem}.wav"
        trimmed.export(intermediate_path, format="wav")
        intermediate_paths.append(intermediate_path)

        removed_ms = sum(e - s for s, e in silence_cuts)
        silence_log.append(
            {
                "segment": segment.stem,
                # Durations are measured against the post-dedup audio that
                # was fed into strip_silence, not the raw on-disk file, so
                # `pre_duration_ms` already reflects any dedup cuts.
                "pre_duration_ms": pre_len_ms,
                "post_duration_ms": len(trimmed),
                "cut_count": len(silence_cuts),
                "cut_total_ms": removed_ms,
                "cuts_ms": [[s, e] for s, e in silence_cuts],
            }
        )
        _console.log(
            f"prepared {segment.stem} ({len(trimmed) / 1000:.1f}s, "
            f"−{len(silence_cuts)} silence cut(s))"
        )

    write_decisions(session, all_spans)
    session.silence_log_path.write_text(json.dumps(silence_log, indent=2))

    intro_path, mid_path = require_jingles()
    tracks = _build_episode_tracks(intro_path, mid_path, intermediate_paths)
    mix_tracks(tracks, session.assembled_path)
    _console.log(
        f"assembled {len(intermediate_paths)} segment(s): intro ducked "
        f"{INTRO_DUCKED_GAIN_DB:+.0f}dB / first topic at "
        f"{INTRO_TOPIC_START_MS / 1000:.1f}s; "
        f"{max(0, len(intermediate_paths) - 1)} mid jingle(s) with "
        f"{MID_PREV_OVERLAP_MS / 1000:.1f}s pre-roll + "
        f"{(MID_NEXT_DELAY_MS - MID_PREV_OVERLAP_MS) / 1000:.1f}s hand-off "
        f"→ {session.assembled_path.name}"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    master_audio(session.assembled_path, output)
    _console.log(f"[green]done[/green] → {output}")
    return output
