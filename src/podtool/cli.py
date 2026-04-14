from __future__ import annotations

from pathlib import Path

import typer

from .dedup import plan_session_cuts, write_decisions
from .io_utils import discover_session
from .master import master_audio
from .pipeline import process_session
from .silence import strip_silence
from .transcribe import transcribe_session

app = typer.Typer(
    name="podtool",
    help="Headless podcast editor: transcribe, dedup, trim, assemble, master.",
    no_args_is_help=True,
)


@app.command()
def process(
    session_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Folder of numbered segment recordings (e.g. 01_intro.wav, 02_topic.wav).",
    ),
    output: Path = typer.Option(
        Path("final.mp3"), "--output", "-o", help="Path for the final mastered MP3."
    ),
    model: str = typer.Option(
        "small",
        "--model",
        "-m",
        help="Faster-Whisper model size (tiny, base, small, medium, large).",
    ),
    dedup_threshold: float = typer.Option(
        0.9,
        "--dedup-threshold",
        help="Text similarity (0-1) above which two windows are treated as the same retake.",
    ),
    no_dedup: bool = typer.Option(
        False, "--no-dedup", help="Skip retake detection and keep every take."
    ),
) -> None:
    """Run the full pipeline on a session folder: transcribe, dedup, strip silence, assemble, master."""
    process_session(
        session_root=session_dir,
        output=output,
        model_size=model,
        dedup_threshold=dedup_threshold,
        dedup_enabled=not no_dedup,
    )


@app.command()
def transcribe(
    session_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Session folder whose segments should be transcribed.",
    ),
    model: str = typer.Option(
        "small",
        "--model",
        "-m",
        help="Faster-Whisper model size (tiny, base, small, medium, large).",
    ),
) -> None:
    """Populate the transcript cache for every segment without touching the audio."""
    session = discover_session(session_dir)
    transcribe_session(session, model_size=model)


@app.command()
def dedup(
    session_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Session folder with an existing transcript cache.",
    ),
    threshold: float = typer.Option(
        0.9,
        "--threshold",
        help="Text similarity (0-1) above which two windows are treated as the same retake.",
    ),
) -> None:
    """Re-run retake detection from cached transcripts and write decisions.json."""
    session = discover_session(session_dir)
    transcripts = transcribe_session(session)
    spans = plan_session_cuts(transcripts, threshold=threshold)
    write_decisions(session, spans)
    typer.echo(f"Wrote {len(spans)} decisions to {session.decisions_path}")


@app.command()
def trim(
    input_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Audio file to silence-trim."
    ),
    output_path: Path = typer.Argument(
        ..., help="Destination file; extension decides the format."
    ),
) -> None:
    """Strip silence from a single audio file (soft trim with padding + crossfades)."""
    from pydub import AudioSegment

    audio = AudioSegment.from_file(input_path)
    trimmed = strip_silence(audio)
    trimmed.export(output_path, format=output_path.suffix.lstrip(".") or "wav")
    typer.echo(
        f"Trimmed {len(audio) / 1000:.2f}s → {len(trimmed) / 1000:.2f}s → {output_path}"
    )


@app.command()
def master(
    input_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Audio file to master (wav/flac/mp3)."
    ),
    output_path: Path = typer.Argument(
        ..., help="Destination file; extension decides the format (.mp3 or .wav)."
    ),
) -> None:
    """Run the mastering chain (gate → HPF → compressor → limiter) on a single audio file."""
    master_audio(input_path, output_path)
    typer.echo(f"Mastered → {output_path}")


if __name__ == "__main__":
    app()
