from podtool.dedup import find_retakes
from podtool.models import TranscriptChunk


def _chunks(*specs: tuple[float, float, str]) -> list[TranscriptChunk]:
    return [TranscriptChunk(start=s, end=e, text=t) for s, e, t in specs]


def test_no_retakes_returns_empty():
    chunks = _chunks(
        (0.0, 1.0, "hello world"),
        (1.0, 2.0, "this is a test"),
        (2.0, 3.0, "of the system"),
        (3.0, 4.0, "we are recording"),
        (4.0, 5.0, "a podcast episode"),
        (5.0, 6.0, "about software"),
        (6.0, 7.0, "engineering practices"),
        (7.0, 8.0, "and tooling decisions"),
        (8.0, 9.0, "across the team"),
        (9.0, 10.0, "in 2026"),
    )
    assert find_retakes("seg", chunks) == []


def test_obvious_retake_detected():
    phrase = ["welcome", "to", "the", "show", "today"]
    chunks = _chunks(
        *((float(i), float(i + 1), word) for i, word in enumerate(phrase)),
        (5.0, 6.0, "uh wait"),
        *((float(6 + i), float(7 + i), word) for i, word in enumerate(phrase)),
        (11.0, 12.0, "and now"),
        (12.0, 13.0, "the topic"),
    )
    spans = find_retakes("seg", chunks, threshold=0.9, window=5)
    assert len(spans) == 1
    span = spans[0]
    assert span.span_start == 0.0
    assert span.span_end == 6.0
    assert span.similarity >= 0.9


def test_short_input_returns_empty():
    chunks = _chunks((0.0, 1.0, "hi"), (1.0, 2.0, "there"))
    assert find_retakes("seg", chunks) == []
