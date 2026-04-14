from __future__ import annotations

SILENCE_THRESH_DB = -40.0
MIN_SILENCE_MS = 400
PAD_MS = 200
CROSSFADE_MS = 10


def strip_silence_with_log(
    audio: "AudioSegment",
) -> tuple["AudioSegment", list[tuple[int, int]]]:
    """Remove silent stretches but leave a small padding around speech and
    crossfade the joins so cuts are inaudible. Returns the trimmed audio and
    a list of `(start_ms, end_ms)` spans in the **original** audio timeline
    that were removed (i.e. the complement of the kept regions)."""
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    original_len = len(audio)
    nonsilent = detect_nonsilent(
        audio,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=SILENCE_THRESH_DB,
    )
    if not nonsilent:
        empty = AudioSegment.silent(duration=0, frame_rate=audio.frame_rate)
        cuts = [(0, original_len)] if original_len > 0 else []
        return empty, cuts

    padded: list[tuple[int, int]] = []
    for start, end in nonsilent:
        padded.append((max(0, start - PAD_MS), min(original_len, end + PAD_MS)))

    merged: list[tuple[int, int]] = [padded[0]]
    for start, end in padded[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    # Complement of the merged keep-regions = the silence spans that got cut.
    cuts: list[tuple[int, int]] = []
    cursor = 0
    for s, e in merged:
        if s > cursor:
            cuts.append((cursor, s))
        cursor = e
    if cursor < original_len:
        cuts.append((cursor, original_len))

    pieces = [audio[s:e] for s, e in merged]
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.append(piece, crossfade=CROSSFADE_MS)
    return out, cuts


def strip_silence(audio: "AudioSegment") -> "AudioSegment":
    """Thin wrapper around `strip_silence_with_log` that discards the cut log."""
    out, _cuts = strip_silence_with_log(audio)
    return out
