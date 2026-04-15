from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .io_utils import is_cache_fresh
from .models import Segment, Session, TranscriptChunk, Word


class _LegacyCacheError(Exception):
    """Raised when a cached transcript predates word-level timestamps."""

_console = Console()
_model_cache: dict[str, object] = {}


def _load_model(model_size: str):
    if model_size in _model_cache:
        return _model_cache[model_size]
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    _model_cache[model_size] = model
    return model


def _transcript_path(session: Session, segment: Segment) -> Path:
    return session.transcripts_dir / f"{segment.stem}.json"


def transcribe_session(session: Session, model_size: str = "small") -> dict[str, list[TranscriptChunk]]:
    """Transcribe every segment in the session, using on-disk cache."""
    results: dict[str, list[TranscriptChunk]] = {}
    pending: list[Segment] = []
    for segment in session.segments:
        cache = _transcript_path(session, segment)
        if is_cache_fresh(cache, segment.path):
            try:
                results[segment.stem] = _load_chunks(cache)
            except _LegacyCacheError:
                _console.log(
                    f"[dim]stale cache[/dim] {segment.stem} "
                    "(pre-word-timestamp format, re-transcribing)"
                )
                pending.append(segment)
            else:
                _console.log(f"[dim]cache hit[/dim] {segment.stem}")
        else:
            pending.append(segment)

    if pending:
        model = _load_model(model_size)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=_console,
        ) as progress:
            task = progress.add_task("transcribing", total=len(pending))
            for segment in pending:
                progress.update(task, description=f"transcribing {segment.stem}")
                chunks = list(_run_whisper(model, segment.path))
                _write_chunks(_transcript_path(session, segment), chunks)
                results[segment.stem] = chunks
                progress.advance(task)

    return results


def _run_whisper(model, audio_path: Path) -> Iterable[TranscriptChunk]:
    segments, _info = model.transcribe(
        str(audio_path), vad_filter=True, word_timestamps=True
    )
    for s in segments:
        words = [
            Word(start=float(w.start), end=float(w.end), text=w.word.strip())
            for w in (s.words or [])
            if w.start is not None and w.end is not None
        ]
        yield TranscriptChunk(
            start=float(s.start),
            end=float(s.end),
            text=s.text.strip(),
            words=words,
        )


def _write_chunks(path: Path, chunks: list[TranscriptChunk]) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "start": c.start,
                    "end": c.end,
                    "text": c.text,
                    "words": [
                        {"start": w.start, "end": w.end, "text": w.text}
                        for w in c.words
                    ],
                }
                for c in chunks
            ],
            indent=2,
        )
    )


def _load_chunks(path: Path) -> list[TranscriptChunk]:
    raw = json.loads(path.read_text())
    chunks: list[TranscriptChunk] = []
    for r in raw:
        if "words" not in r:
            raise _LegacyCacheError(str(path))
        words = [
            Word(start=w["start"], end=w["end"], text=w["text"]) for w in r["words"]
        ]
        chunks.append(
            TranscriptChunk(
                start=r["start"], end=r["end"], text=r["text"], words=words
            )
        )
    return chunks
