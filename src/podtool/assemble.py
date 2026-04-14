from __future__ import annotations

from pathlib import Path

CROSSFADE_MS = 10


def assemble(intermediate_paths: list[Path], output_path: Path) -> Path:
    """Concatenate WAVs in the given order with a short crossfade and write
    a single WAV to `output_path`."""
    from pydub import AudioSegment

    if not intermediate_paths:
        raise ValueError("No segments to assemble")

    out = AudioSegment.from_file(intermediate_paths[0])
    for path in intermediate_paths[1:]:
        piece = AudioSegment.from_file(path)
        out = out.append(piece, crossfade=CROSSFADE_MS)

    out.export(output_path, format="wav")
    return output_path
