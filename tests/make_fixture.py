"""Generate a synthetic session folder for the smoke test.

Creates four 'segment' WAVs at /tmp/podtool-fixture/:
  01_intro.wav     — sine A (speech-stand-in) + silence
  02_topic_a.wav   — sine B with a deliberate retake (same window twice)
  03_mid_tune.wav  — sine C
  04_topic_b.wav   — sine D + trailing silence
"""

from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine


def tone(freq: int, ms: int, gain_db: float = -6.0) -> AudioSegment:
    return Sine(freq).to_audio_segment(duration=ms).apply_gain(gain_db)


def silence(ms: int) -> AudioSegment:
    return AudioSegment.silent(duration=ms, frame_rate=44100)


def main() -> None:
    out = Path("/tmp/podtool-fixture")
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.wav"):
        f.unlink()

    # 01: tone + long silence + tone (silence stripper should kill the gap)
    intro = tone(440, 600) + silence(1500) + tone(440, 600)
    intro.export(out / "01_intro.wav", format="wav")

    # 02: same tone block twice in a row (a retake) — dedup may or may not
    # catch this depending on whisper transcription, but the audio still
    # round-trips through every stage.
    block = tone(660, 800) + silence(200)
    topic_a = block + block + tone(660, 400)
    topic_a.export(out / "02_topic_a.wav", format="wav")

    # 03: short musical interlude
    mid = tone(880, 1000)
    mid.export(out / "03_mid_tune.wav", format="wav")

    # 04: another talk segment with trailing silence
    topic_b = tone(550, 1000) + silence(2000)
    topic_b.export(out / "04_topic_b.wav", format="wav")

    print(f"wrote 4 segments to {out}")


if __name__ == "__main__":
    main()
