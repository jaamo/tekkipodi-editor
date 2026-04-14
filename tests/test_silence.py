import pytest

pydub = pytest.importorskip("pydub")

from podtool.silence import strip_silence


def _tone(duration_ms: int, freq: int = 440):
    from pydub.generators import Sine

    return Sine(freq).to_audio_segment(duration=duration_ms).apply_gain(-3)


def _silence(duration_ms: int):
    from pydub import AudioSegment

    return AudioSegment.silent(duration=duration_ms, frame_rate=44100)


def test_strip_removes_long_silence():
    audio = _tone(800) + _silence(1500) + _tone(800)
    original_len = len(audio)
    out = strip_silence(audio)
    assert len(out) < original_len
    # both tones should still be there (with padding)
    assert len(out) >= 1600


def test_strip_keeps_speech_only():
    audio = _silence(1000) + _tone(500) + _silence(1000)
    out = strip_silence(audio)
    assert len(out) < len(audio)
    assert len(out) >= 500
