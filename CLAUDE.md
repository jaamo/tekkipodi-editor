# CLAUDE.md

Project-specific guidance for Claude Code. See `README.md` for end-user usage.

## Environment

- Python is managed by **uv** in a project-local `.venv/` (CPython 3.12).
  Always run tools via `.venv/bin/python` or with `VIRTUAL_ENV=.venv uv pip ...`.
  Do not fall back to `/usr/bin/python3` — it's a different Python (3.14) with a
  separately-installed, easily-drifting copy of the deps.
- No `uv.lock` exists. To sync deps after a `pyproject.toml` change, run
  `uv pip install -e .` with the venv active.
- System dep: `ffmpeg` (used via `pydub`).

## Architecture

Pipeline (`src/podtool/pipeline.py::process_session`) runs in a fixed order:

1. `transcribe.py` — Faster-Whisper per segment, cached on disk by mtime.
2. `dedup.py` — sliding 5-chunk window + rapidfuzz `fuzz.ratio` similarity,
   60 s proximity; when a near-duplicate is found, the **earlier** take is cut
   (keep-the-last-take rule).
3. `silence.py` — `pydub.detect_nonsilent` + padding + 10 ms crossfades.
4. `assemble.py` — `mix_tracks(tracks, out)` mixes a list of
   `(AudioSegment, position_ms)` onto a silent base and writes WAV. The
   pipeline builds that track list in `pipeline._build_episode_tracks` so
   `assemble.py` stays dumb — it only knows how to overlay. (The older
   `assemble(paths, out)` concat API is still exported for the assemble
   test and as a plain-concat primitive, but the pipeline no longer calls
   it.)
5. `master.py` — pyloudnorm gain-match to −16 LUFS, then a Pedalboard chain
   (NoiseGate → HighpassFilter → Limiter). Output format from extension;
   `.mp3` takes a separate pydub path.

Heavy deps (`pedalboard`, `pyloudnorm`, `faster_whisper`, `pydub`, `numpy`)
are imported **inside functions**, not at module top. Preserve this — it keeps
`podtool --help` fast and avoids pulling CUDA/torch on simple commands.

## Session layout

A session is just a directory of numbered audio files (`01_*.wav`,
`02_*.wav`, ...). `discover_session` sorts lexicographically — the numeric
prefix is the ordering contract.

Per-session cache (auto-created) lives under `<session>/.podtool/`:

```
.podtool/
├── transcripts/<stem>.json    # faster-whisper output, mtime-invalidated
├── intermediate/<stem>.wav    # post-dedup, post-silence-strip
├── decisions.json             # dedup CutSpans that were applied
├── silence_log.json           # per-segment silence spans removed by strip_silence
└── assembled.wav              # pre-master concat
```

Transcript cache freshness is an mtime compare (`io_utils.is_cache_fresh`):
touching a source file re-triggers transcription for that segment only.

## Jingle convention

`intro.wav` and `mid.wav` at the **repo root** are hard-coded in
`io_utils.py` (`INTRO_JINGLE`, `MID_JINGLE`). Both jingles are attenuated by
`JINGLE_GAIN_DB` (−9 dB) before anything else happens to them, so they sit
under the speech; tune that single constant in `pipeline.py` to make the
jingles louder or quieter episode-wide. Both jingles also participate in
cross-plays with adjacent segments; the timings live as module-level
constants at the top of `pipeline.py`.

**Intro (opening):** the intro plays at full volume, then ducks linearly by
`INTRO_DUCKED_GAIN_DB` (−10 dB) across
`[INTRO_DUCK_START_MS, INTRO_DUCK_END_MS]` (5–7 s), and the first topic
starts at `INTRO_TOPIC_START_MS` (5.8 s) on top of the now-ducked intro.
The intro then sits at the ducked level under the topic until
`INTRO_FADEOUT_START_MS` (10 s), at which point it fades linearly to silence
by `INTRO_FADEOUT_END_MS` (15 s) and is truncated there (anything past the
fade-out end is discarded). The first topic continues solo after the intro
is gone. Both the duck and the fade-out are applied by `_duck_intro()`
before the intro is placed on the timeline.

**Mid jingle (between topics):** each mid jingle starts
`MID_PREV_OVERLAP_MS` (1.4 s) before the previous segment ends and the next
segment starts `MID_NEXT_DELAY_MS` (4.0 s) after the jingle began — so
1.4 s of jingle-over-previous-segment overlap, followed by
(MID_NEXT_DELAY_MS − MID_PREV_OVERLAP_MS) = 2.6 s of mid-solo, followed by
the next segment entering while the mid's tail is still playing (no duck).

The whole episode is assembled as one pass: `_build_episode_tracks()`
returns a list of `(AudioSegment, absolute_position_ms)` tuples covering the
ducked intro, every segment, and every mid jingle, and `assemble.mix_tracks`
overlays them onto a silent base sized to fit the lot.

`require_jingles()` raises if either file is missing. There is no CLI flag
to override — changing jingles means replacing the files at the repo root
or editing the `INTRO_*` / `MID_*` constants.

## Output format

Default output is WAV (`final.wav`). MP3 works via the `_write_mp3` branch
in `master.py` but is not the preferred path; prefer WAV unless the user
explicitly asks for MP3.

## Tests

`pytest` under `tests/`. `tests/make_fixture.py` generates synthetic audio
fixtures — use it (don't commit real audio) when adding tests that need
input files.

## When editing

- New pipeline stages go through `pipeline.py`, not the CLI. The CLI in
  `cli.py` is thin — each subcommand is a user-facing entry into one stage.
- Tuning knobs live as module-level constants (`TARGET_LUFS`,
  `SILENCE_THRESH_DB`, `WINDOW_CHUNKS`, `PROXIMITY_SECONDS`, ...). Prefer
  adjusting those over adding new CLI flags unless the user asks.
