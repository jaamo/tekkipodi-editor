from __future__ import annotations

from pathlib import Path

from rich.console import Console

from .assemble import assemble
from .dedup import apply_cuts, find_retakes, write_decisions
from .io_utils import discover_session, require_jingles
from .master import master_audio
from .silence import strip_silence
from .transcribe import transcribe_session

_console = Console()


def process_session(
    session_root: Path,
    output: Path,
    model_size: str = "small",
    dedup_threshold: float = 0.9,
    dedup_enabled: bool = True,
) -> Path:
    session = discover_session(session_root)
    _console.rule(f"[bold]podtool[/bold] {session_root.name}")
    _console.log(f"{len(session.segments)} segments found")

    transcripts = transcribe_session(session, model_size=model_size)

    all_spans = []
    intermediate_paths: list[Path] = []
    for segment in session.segments:
        chunks = transcripts[segment.stem]
        spans = (
            find_retakes(segment.stem, chunks, threshold=dedup_threshold)
            if dedup_enabled
            else []
        )
        all_spans.extend(spans)

        from pydub import AudioSegment

        if spans:
            audio = apply_cuts(segment.path, spans)
            _console.log(f"dedup {segment.stem}: removed {len(spans)} span(s)")
        else:
            audio = AudioSegment.from_file(segment.path)

        trimmed = strip_silence(audio)
        intermediate_path = session.intermediate_dir / f"{segment.stem}.wav"
        trimmed.export(intermediate_path, format="wav")
        intermediate_paths.append(intermediate_path)
        _console.log(
            f"prepared {segment.stem} ({len(trimmed) / 1000:.1f}s)"
        )

    write_decisions(session, all_spans)

    intro, mid = require_jingles()
    timeline: list[Path] = [intro]
    for i, seg_path in enumerate(intermediate_paths):
        if i > 0:
            timeline.append(mid)
        timeline.append(seg_path)
    _console.log(
        f"timeline: intro + {len(intermediate_paths)} segment(s) with "
        f"{max(0, len(intermediate_paths) - 1)} mid jingle(s)"
    )

    assemble(timeline, session.assembled_path)
    _console.log(f"assembled → {session.assembled_path.name}")

    output.parent.mkdir(parents=True, exist_ok=True)
    master_audio(session.assembled_path, output)
    _console.log(f"[green]done[/green] → {output}")
    return output
