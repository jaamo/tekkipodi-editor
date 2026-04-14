from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydub import AudioSegment

CROSSFADE_MS = 10


def mix_tracks(
    tracks: list[tuple["AudioSegment", int]], output_path: Path
) -> Path:
    """Mix a list of `(audio, position_ms)` tracks onto a silent base sized
    to hold them all and write the result to `output_path` as WAV. Unlike
    `assemble`, tracks may overlap in time — each track is layered on top of
    whatever is already at that position via pydub's `overlay`."""
    from pydub import AudioSegment

    if not tracks:
        raise ValueError("No tracks to mix")

    first_audio = tracks[0][0]
    fr = first_audio.frame_rate
    ch = first_audio.channels

    total_len = max(pos + len(audio) for audio, pos in tracks)
    base = AudioSegment.silent(duration=total_len, frame_rate=fr).set_channels(ch)
    for audio, pos in tracks:
        base = base.overlay(audio, position=pos)

    base.export(output_path, format="wav")
    return output_path


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
