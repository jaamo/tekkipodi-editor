from __future__ import annotations

from pathlib import Path

TARGET_LUFS = -16.0
PEAK_CEILING_DB = -1.0


def build_chain():
    from pedalboard import HighpassFilter, Limiter, NoiseGate, Pedalboard

    return Pedalboard(
        [
            NoiseGate(threshold_db=-40),
            HighpassFilter(cutoff_frequency_hz=80),
            Limiter(threshold_db=PEAK_CEILING_DB, release_ms=100),
        ]
    )


def master_audio(input_path: Path, output_path: Path) -> Path:
    """Run the mastering chain and write the result to `output_path`. The
    output format is inferred from the file extension; MP3 is encoded via
    pydub/ffmpeg since pedalboard's MP3 writer is not available everywhere."""
    import math

    import numpy as np
    import pyloudnorm as pyln
    from pedalboard.io import AudioFile

    with AudioFile(str(input_path)) as f:
        samplerate = f.samplerate
        audio = f.read(f.frames)

    # pyloudnorm expects (frames, channels) float64; pedalboard hands us
    # (channels, frames) float32. Transpose, measure, gain-match to target,
    # transpose back before feeding the pedalboard chain.
    as_fc = np.asarray(audio).T.astype(np.float64, copy=False)
    meter = pyln.Meter(samplerate)
    measured = meter.integrated_loudness(as_fc)
    if math.isfinite(measured):
        as_fc = pyln.normalize.loudness(as_fc, measured, TARGET_LUFS)
    loud = as_fc.T.astype(np.float32, copy=False)

    board = build_chain()
    effected = board(loud, samplerate)

    suffix = output_path.suffix.lower()
    if suffix == ".mp3":
        _write_mp3(output_path, effected, samplerate, effected.shape[0])
    else:
        with AudioFile(str(output_path), "w", samplerate, effected.shape[0]) as f:
            f.write(effected)
    return output_path


def _write_mp3(path: Path, audio, samplerate: int, num_channels: int) -> None:
    """Encode a numpy float32 array to MP3 via pydub."""
    import numpy as np
    from pydub import AudioSegment

    arr = np.asarray(audio)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    arr = np.clip(arr, -1.0, 1.0)
    int16 = (arr * 32767.0).astype(np.int16)
    interleaved = int16.T.reshape(-1)

    segment = AudioSegment(
        interleaved.tobytes(),
        frame_rate=samplerate,
        sample_width=2,
        channels=int16.shape[0],
    )
    segment.export(path, format="mp3", bitrate="192k")
