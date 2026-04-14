from pathlib import Path

import pytest

pydub = pytest.importorskip("pydub")

from podtool.assemble import assemble
from podtool.io_utils import discover_session


def _write_silence(path: Path, ms: int) -> None:
    from pydub import AudioSegment

    AudioSegment.silent(duration=ms, frame_rate=44100).export(path, format="wav")


def test_discover_sorts_by_filename(tmp_path: Path):
    for name in ["03_c.wav", "01_a.wav", "02_b.wav"]:
        _write_silence(tmp_path / name, 100)
    session = discover_session(tmp_path)
    assert [s.stem for s in session.segments] == ["01_a", "02_b", "03_c"]


def test_discover_ignores_dotfiles_and_non_audio(tmp_path: Path):
    _write_silence(tmp_path / "01_a.wav", 100)
    (tmp_path / "notes.txt").write_text("hi")
    (tmp_path / ".hidden.wav").write_bytes(b"")
    session = discover_session(tmp_path)
    assert [s.stem for s in session.segments] == ["01_a"]


def test_assemble_concatenates_in_order(tmp_path: Path):
    from pydub import AudioSegment

    paths = []
    for i, ms in enumerate([200, 300, 400], start=1):
        p = tmp_path / f"0{i}.wav"
        _write_silence(p, ms)
        paths.append(p)
    out_path = tmp_path / "out.wav"
    assemble(paths, out_path)
    out = AudioSegment.from_file(out_path)
    # 200 + 300 + 400 minus two 10ms crossfades
    assert 870 <= len(out) <= 900
