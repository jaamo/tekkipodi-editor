from __future__ import annotations

import json
from pathlib import Path

from rapidfuzz import fuzz

from .models import CutSpan, Session, TranscriptChunk

WINDOW_CHUNKS = 5
PROXIMITY_SECONDS = 60.0


def find_retakes(
    segment_name: str,
    chunks: list[TranscriptChunk],
    threshold: float = 0.9,
    window: int = WINDOW_CHUNKS,
    proximity: float = PROXIMITY_SECONDS,
) -> list[CutSpan]:
    """Slide a window over chunks; when a near-duplicate window appears within
    `proximity` seconds, mark the earlier occurrence for deletion (keep the
    last take). Returns non-overlapping spans sorted by start time."""
    if len(chunks) < window * 2:
        return []

    score_cutoff = threshold * 100
    spans: list[CutSpan] = []
    consumed_until = -1.0

    for i in range(len(chunks) - window):
        if chunks[i].start < consumed_until:
            continue
        text_a = " ".join(c.text for c in chunks[i : i + window])
        if not text_a.strip():
            continue
        best: tuple[int, float] | None = None
        for j in range(i + window, len(chunks) - window + 1):
            if chunks[j].start - chunks[i].start > proximity:
                break
            text_b = " ".join(c.text for c in chunks[j : j + window])
            score = fuzz.ratio(text_a, text_b)
            if score >= score_cutoff and (best is None or score > best[1]):
                best = (j, score)
        if best is not None:
            j, score = best
            span_start = chunks[i].start
            span_end = chunks[j].start
            kept_text = " ".join(c.text for c in chunks[j : j + window])
            spans.append(
                CutSpan(
                    segment=segment_name,
                    span_start=span_start,
                    span_end=span_end,
                    similarity=score / 100.0,
                    kept_text=kept_text,
                )
            )
            consumed_until = span_end

    return spans


def plan_session_cuts(
    transcripts: dict[str, list[TranscriptChunk]],
    threshold: float = 0.9,
) -> list[CutSpan]:
    cuts: list[CutSpan] = []
    for stem, chunks in transcripts.items():
        cuts.extend(find_retakes(stem, chunks, threshold=threshold))
    return cuts


def write_decisions(session: Session, spans: list[CutSpan]) -> None:
    session.decisions_path.write_text(
        json.dumps([s.to_dict() for s in spans], indent=2)
    )


def apply_cuts(audio_path: Path, spans: list[CutSpan]) -> "AudioSegment":
    """Return a pydub AudioSegment with the given spans removed."""
    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path)
    if not spans:
        return audio

    spans_sorted = sorted(spans, key=lambda s: s.span_start)
    keep_pieces: list[AudioSegment] = []
    cursor_ms = 0
    for span in spans_sorted:
        start_ms = int(span.span_start * 1000)
        end_ms = int(span.span_end * 1000)
        if start_ms > cursor_ms:
            keep_pieces.append(audio[cursor_ms:start_ms])
        cursor_ms = max(cursor_ms, end_ms)
    if cursor_ms < len(audio):
        keep_pieces.append(audio[cursor_ms:])

    if not keep_pieces:
        return AudioSegment.silent(duration=0, frame_rate=audio.frame_rate)

    out = keep_pieces[0]
    for piece in keep_pieces[1:]:
        out = out.append(piece, crossfade=10)
    return out
