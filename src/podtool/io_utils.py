from __future__ import annotations

from pathlib import Path

from .models import Segment, Session

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aiff", ".aif"}

# Repo root containing pyproject.toml, intro.wav, mid.wav.
# src/podtool/io_utils.py -> src/podtool -> src -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]
INTRO_JINGLE = REPO_ROOT / "intro.wav"
MID_JINGLE = REPO_ROOT / "mid.wav"
TONE_OF_VOICE = REPO_ROOT / "tone_of_voice.md"


def require_jingles() -> tuple[Path, Path]:
    """Return (intro, mid) paths; raise if either is missing."""
    missing = [p for p in (INTRO_JINGLE, MID_JINGLE) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing jingle(s): " + ", ".join(str(p) for p in missing)
        )
    return INTRO_JINGLE, MID_JINGLE


def require_tone_of_voice() -> Path:
    """Return the tone-of-voice file path; raise if it is missing."""
    if not TONE_OF_VOICE.exists():
        raise FileNotFoundError(
            f"Tone-of-voice file not found: {TONE_OF_VOICE}. "
            "Create it at the repo root before running `podtool blog`."
        )
    return TONE_OF_VOICE


def discover_session(root: Path) -> Session:
    """Find audio segments in `root`, sorted by filename."""
    if not root.is_dir():
        raise NotADirectoryError(f"Session path is not a directory: {root}")

    files = sorted(
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS and not p.name.startswith(".")
    )
    if not files:
        raise FileNotFoundError(f"No audio segments found in {root}")

    session = Session(root=root, segments=[Segment.from_path(p) for p in files])
    session.cache_dir.mkdir(exist_ok=True)
    session.transcripts_dir.mkdir(exist_ok=True)
    session.intermediate_dir.mkdir(exist_ok=True)
    return session


def is_cache_fresh(cache_path: Path, source_path: Path) -> bool:
    if not cache_path.exists():
        return False
    return cache_path.stat().st_mtime >= source_path.stat().st_mtime
