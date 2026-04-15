import pytest

from podtool.dedup import (
    _merge_close_spans,
    _tokenize_strikethrough,
    parse_review_markdown,
    write_review_markdown,
)
from podtool.models import CutSpan, Session, TranscriptChunk, Word


def _chunk(start: float, end: float, words: list[tuple[str, float, float]]) -> TranscriptChunk:
    return TranscriptChunk(
        start=start,
        end=end,
        text=" ".join(w[0] for w in words),
        words=[Word(start=s, end=e, text=t) for t, s, e in words],
    )


def test_tokenize_no_strikethrough():
    assert _tokenize_strikethrough("hello there world") == [
        ("hello", False),
        ("there", False),
        ("world", False),
    ]


def test_tokenize_single_word_struck():
    assert _tokenize_strikethrough("keep ~~cut~~ keep") == [
        ("keep", False),
        ("cut", True),
        ("keep", False),
    ]


def test_tokenize_phrase_struck():
    assert _tokenize_strikethrough("hi ~~cut this phrase~~ end") == [
        ("hi", False),
        ("cut", True),
        ("this", True),
        ("phrase", True),
        ("end", False),
    ]


def test_tokenize_multiple_strikethrough_regions():
    assert _tokenize_strikethrough("~~a~~ b ~~c d~~ e") == [
        ("a", True),
        ("b", False),
        ("c", True),
        ("d", True),
        ("e", False),
    ]


def test_tokenize_whole_line_struck():
    assert _tokenize_strikethrough("~~all of this~~") == [
        ("all", True),
        ("of", True),
        ("this", True),
    ]


def test_merge_close_spans_merges_adjacent():
    spans = [
        CutSpan(segment="seg", span_start=1.0, span_end=2.0, similarity=1.0, kept_text=""),
        CutSpan(segment="seg", span_start=2.01, span_end=3.0, similarity=1.0, kept_text=""),
    ]
    merged = _merge_close_spans(spans)
    assert len(merged) == 1
    assert merged[0].span_start == 1.0
    assert merged[0].span_end == 3.0


def test_merge_close_spans_keeps_far_apart():
    spans = [
        CutSpan(segment="seg", span_start=1.0, span_end=2.0, similarity=1.0, kept_text=""),
        CutSpan(segment="seg", span_start=3.0, span_end=4.0, similarity=1.0, kept_text=""),
    ]
    assert _merge_close_spans(spans) == spans


def test_roundtrip_no_edits_yields_no_spans(tmp_path):
    session = Session(root=tmp_path, segments=[])
    session.cache_dir.mkdir()
    transcripts = {
        "01_intro": [
            _chunk(0.0, 2.0, [("hello", 0.0, 0.5), ("world", 0.5, 1.0), ("foo", 1.0, 2.0)]),
            _chunk(2.0, 4.0, [("bar", 2.0, 2.5), ("baz", 2.5, 4.0)]),
        ],
    }
    write_review_markdown(session, transcripts, auto_spans=[])
    spans = parse_review_markdown(session, transcripts)
    assert spans == []


def test_roundtrip_preserves_auto_flagged_cuts(tmp_path):
    session = Session(root=tmp_path, segments=[])
    session.cache_dir.mkdir()
    chunk = _chunk(0.0, 2.0, [("hello", 0.0, 0.5), ("world", 0.5, 1.0), ("foo", 1.0, 2.0)])
    transcripts = {"01_intro": [chunk]}
    auto = [
        CutSpan(
            segment="01_intro",
            span_start=0.0,
            span_end=2.0,
            similarity=0.95,
            kept_text="",
        )
    ]
    write_review_markdown(session, transcripts, auto_spans=auto)
    spans = parse_review_markdown(session, transcripts)
    assert len(spans) == 1
    assert spans[0].segment == "01_intro"
    assert spans[0].span_start == 0.0
    assert spans[0].span_end == 2.0


def test_user_can_add_midline_strikethrough(tmp_path):
    session = Session(root=tmp_path, segments=[])
    session.cache_dir.mkdir()
    chunk = _chunk(
        0.0,
        2.0,
        [
            ("keep", 0.0, 0.3),
            ("cut", 0.3, 0.6),
            ("this", 0.6, 0.9),
            ("keep", 0.9, 2.0),
        ],
    )
    transcripts = {"01_intro": [chunk]}
    write_review_markdown(session, transcripts, auto_spans=[])

    # Simulate a user hand-edit: strike words 1–2 ("cut this") inside the bullet.
    original = session.review_path.read_text()
    edited = original.replace(
        "- [0.00–2.00] keep cut this keep",
        "- [0.00–2.00] keep ~~cut this~~ keep",
    )
    assert edited != original, "edit pattern did not match the rendered bullet"
    session.review_path.write_text(edited)

    spans = parse_review_markdown(session, transcripts)
    assert len(spans) == 1
    assert spans[0].span_start == 0.3
    assert spans[0].span_end == 0.9


def test_word_count_mismatch_raises(tmp_path):
    session = Session(root=tmp_path, segments=[])
    session.cache_dir.mkdir()
    chunk = _chunk(0.0, 2.0, [("hello", 0.0, 1.0), ("world", 1.0, 2.0)])
    transcripts = {"01_intro": [chunk]}
    write_review_markdown(session, transcripts, auto_spans=[])
    # User added a word that wasn't in the original transcript.
    bad = session.review_path.read_text().replace(
        "hello world", "hello there world"
    )
    session.review_path.write_text(bad)

    with pytest.raises(ValueError, match="word count mismatch"):
        parse_review_markdown(session, transcripts)


def test_contiguous_strikethrough_across_chunks_merges(tmp_path):
    session = Session(root=tmp_path, segments=[])
    session.cache_dir.mkdir()
    # Two chunks whose word timestamps touch: first ends at 1.0, second
    # starts at 1.0. Strikethrough at the end of chunk 1 and start of
    # chunk 2 should merge across the boundary.
    c1 = _chunk(0.0, 1.0, [("keep", 0.0, 0.5), ("cut", 0.5, 1.0)])
    c2 = _chunk(1.0, 2.0, [("cut", 1.0, 1.5), ("keep", 1.5, 2.0)])
    transcripts = {"01_intro": [c1, c2]}
    write_review_markdown(session, transcripts, auto_spans=[])

    text = session.review_path.read_text()
    text = text.replace(
        "- [0.00–1.00] keep cut", "- [0.00–1.00] keep ~~cut~~"
    )
    text = text.replace(
        "- [1.00–2.00] cut keep", "- [1.00–2.00] ~~cut~~ keep"
    )
    session.review_path.write_text(text)

    spans = parse_review_markdown(session, transcripts)
    assert len(spans) == 1
    assert spans[0].span_start == 0.5
    assert spans[0].span_end == 1.5
