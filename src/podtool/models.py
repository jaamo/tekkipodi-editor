from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    path: Path
    stem: str

    @classmethod
    def from_path(cls, path: Path) -> "Segment":
        return cls(path=path, stem=path.stem)


@dataclass(frozen=True)
class TranscriptChunk:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class CutSpan:
    """A range of audio (seconds) to remove from a segment."""

    segment: str
    span_start: float
    span_end: float
    similarity: float
    kept_text: str

    def to_dict(self) -> dict:
        return {
            "segment": self.segment,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "similarity": self.similarity,
            "kept_text": self.kept_text,
        }


@dataclass
class Session:
    root: Path
    segments: list[Segment] = field(default_factory=list)

    @property
    def cache_dir(self) -> Path:
        return self.root / ".podtool"

    @property
    def transcripts_dir(self) -> Path:
        return self.cache_dir / "transcripts"

    @property
    def intermediate_dir(self) -> Path:
        return self.cache_dir / "intermediate"

    @property
    def decisions_path(self) -> Path:
        return self.cache_dir / "decisions.json"

    @property
    def silence_log_path(self) -> Path:
        return self.cache_dir / "silence_log.json"

    @property
    def assembled_path(self) -> Path:
        return self.cache_dir / "assembled.wav"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"
