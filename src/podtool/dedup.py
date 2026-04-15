from __future__ import annotations

import json
import re
from pathlib import Path

from rapidfuzz import fuzz

from .models import CutSpan, Session, TranscriptChunk

WINDOW_CHUNKS = 5
PROXIMITY_SECONDS = 60.0

# Spans closer than this many seconds are merged so apply_cuts doesn't stack
# two adjacent crossfades on a single user-selected phrase.
MERGE_EPSILON_SECONDS = 0.03


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


_BULLET_RE = re.compile(r"^- \[(\d+\.\d+)[–-](\d+\.\d+)\]\s+(.*)$")
_HEADER_RE = re.compile(r"^## (.+)$")


def _chunk_auto_struck(
    chunk: TranscriptChunk, auto_spans: list[CutSpan]
) -> bool:
    """A chunk is auto-struck if any find_retakes() span overlaps it at all."""
    for span in auto_spans:
        if span.span_start < chunk.end and span.span_end > chunk.start:
            return True
    return False


def _render_chunk_text(chunk: TranscriptChunk, struck: bool) -> str:
    """Render a chunk's word list as the bullet text. If `struck` is True,
    wrap the whole thing in ~~...~~ so every word in the chunk is marked
    for removal on parse-back."""
    if chunk.words:
        body = " ".join(w.text for w in chunk.words)
    else:
        body = chunk.text
    if struck and body:
        return f"~~{body}~~"
    return body


def write_review_markdown(
    session: Session,
    transcripts: dict[str, list[TranscriptChunk]],
    auto_spans: list[CutSpan],
) -> None:
    """Write the human-in-the-loop review file at <session>/review.md.

    Each Whisper chunk becomes one bullet tagged with [start–end] seconds.
    Any chunk that `find_retakes()` flagged is pre-wrapped in ~~...~~. The
    user edits the file in their editor and then re-runs the pipeline;
    `parse_review_markdown` turns the edited file back into word-aligned
    CutSpans.
    """
    by_segment: dict[str, list[CutSpan]] = {}
    for span in auto_spans:
        by_segment.setdefault(span.segment, []).append(span)

    lines: list[str] = [
        f"# Dedup review: {session.root.name}",
        "",
        "Edit this file to mark audio for removal. Wrap any span of words",
        "in `~~...~~` to cut that audio. Strikethrough can be a whole line,",
        "a phrase mid-line, or a single word — cuts land on word boundaries.",
        "Auto-flagged retakes are already marked; un-strike any you want to",
        "keep. Save and return to the terminal to continue. Do NOT add,",
        "delete, or retype words — only add/remove `~~` markers around",
        "existing text.",
        "",
    ]
    for stem, chunks in transcripts.items():
        seg_spans = by_segment.get(stem, [])
        lines.append(f"## {stem}")
        lines.append("")
        for chunk in chunks:
            struck = _chunk_auto_struck(chunk, seg_spans)
            body = _render_chunk_text(chunk, struck)
            lines.append(f"- [{chunk.start:.2f}–{chunk.end:.2f}] {body}")
        lines.append("")

    session.review_path.write_text("\n".join(lines))


def _tokenize_strikethrough(text: str) -> list[tuple[str, bool]]:
    """Split `text` into (word, struck) pairs. `~~` toggles struck state."""
    parts = text.split("~~")
    tokens: list[tuple[str, bool]] = []
    struck = False
    for part in parts:
        for word in part.split():
            tokens.append((word, struck))
        struck = not struck
    return tokens


def _merge_close_spans(spans: list[CutSpan]) -> list[CutSpan]:
    """Merge consecutive CutSpans inside the same segment whose inter-gap
    is smaller than MERGE_EPSILON_SECONDS."""
    if not spans:
        return []
    by_segment: dict[str, list[CutSpan]] = {}
    for s in spans:
        by_segment.setdefault(s.segment, []).append(s)

    merged: list[CutSpan] = []
    for segment, seg_spans in by_segment.items():
        seg_spans.sort(key=lambda s: s.span_start)
        current = seg_spans[0]
        for nxt in seg_spans[1:]:
            if nxt.span_start - current.span_end <= MERGE_EPSILON_SECONDS:
                current = CutSpan(
                    segment=segment,
                    span_start=current.span_start,
                    span_end=max(current.span_end, nxt.span_end),
                    similarity=1.0,
                    kept_text="",
                )
            else:
                merged.append(current)
                current = nxt
        merged.append(current)
    return merged


def parse_review_markdown(
    session: Session,
    transcripts: dict[str, list[TranscriptChunk]],
) -> list[CutSpan]:
    """Read the edited review.md and return word-aligned CutSpans.

    Raises ValueError (with specific segment/time) if a bullet's word count
    drifted from the original transcript — the user added, removed, or
    retyped words instead of just moving `~~` markers.
    """
    text = session.review_path.read_text()

    chunk_index: dict[tuple[str, float], TranscriptChunk] = {}
    for stem, chunks in transcripts.items():
        for chunk in chunks:
            chunk_index[(stem, round(chunk.start, 2))] = chunk

    spans: list[CutSpan] = []
    current_segment: str | None = None
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip()
        header = _HEADER_RE.match(line)
        if header:
            current_segment = header.group(1).strip()
            continue
        bullet = _BULLET_RE.match(line)
        if not bullet:
            continue
        if current_segment is None:
            raise ValueError(
                f"review.md:{line_no}: bullet found before any '## segment' header"
            )
        start = float(bullet.group(1))
        body = bullet.group(3)
        chunk = chunk_index.get((current_segment, round(start, 2)))
        if chunk is None:
            raise ValueError(
                f"review.md:{line_no}: no transcript chunk for "
                f"{current_segment} @ {start:.2f}s — did you edit a "
                "timestamp or segment header?"
            )

        tokens = _tokenize_strikethrough(body)
        if len(tokens) != len(chunk.words):
            raise ValueError(
                f"review.md:{line_no}: word count mismatch in "
                f"{current_segment} @ {start:.2f}s "
                f"(expected {len(chunk.words)}, got {len(tokens)}). "
                "Only add or remove `~~` markers — do not retype, add, "
                "or delete words."
            )

        run_start: int | None = None
        for idx, (_tok, struck) in enumerate(tokens):
            if struck and run_start is None:
                run_start = idx
            elif not struck and run_start is not None:
                spans.append(
                    CutSpan(
                        segment=current_segment,
                        span_start=chunk.words[run_start].start,
                        span_end=chunk.words[idx - 1].end,
                        similarity=1.0,
                        kept_text="",
                    )
                )
                run_start = None
        if run_start is not None and tokens:
            spans.append(
                CutSpan(
                    segment=current_segment,
                    span_start=chunk.words[run_start].start,
                    span_end=chunk.words[len(tokens) - 1].end,
                    similarity=1.0,
                    kept_text="",
                )
            )

    return _merge_close_spans(spans)


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
