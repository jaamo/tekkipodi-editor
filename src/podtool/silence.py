from __future__ import annotations

SILENCE_THRESH_DB = -40.0
MIN_SILENCE_MS = 400
PAD_MS = 200
CROSSFADE_MS = 10


def strip_silence(audio: "AudioSegment") -> "AudioSegment":
    """Remove silent stretches but leave a small padding around speech and
    crossfade the joins so cuts are inaudible."""
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    nonsilent = detect_nonsilent(
        audio,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=SILENCE_THRESH_DB,
    )
    if not nonsilent:
        return AudioSegment.silent(duration=0, frame_rate=audio.frame_rate)

    padded: list[tuple[int, int]] = []
    for start, end in nonsilent:
        padded.append((max(0, start - PAD_MS), min(len(audio), end + PAD_MS)))

    merged: list[tuple[int, int]] = [padded[0]]
    for start, end in padded[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    pieces = [audio[s:e] for s, e in merged]
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.append(piece, crossfade=CROSSFADE_MS)
    return out
