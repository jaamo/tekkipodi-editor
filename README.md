# tekkipodi-editor

Headless CLI that turns a folder of raw podcast segment recordings into a
finished, mastered MP3.

## Pipeline

1. **Transcribe** every segment with Faster-Whisper (cached on disk).
2. **Dedup** probable retakes by text similarity, keeping the last take.
3. **Strip silence** with soft trim + crossfades.
4. **Assemble** segments in filename-prefix order.
5. **Master** through a Pedalboard chain (gate → HPF → compressor → limiter).
6. **Export** the final MP3.

## Install

```bash
pip install -e .
# system dep:
sudo dnf install ffmpeg   # or apt/brew equivalent
```

## Usage

```bash
podtool process ./my-podcast-ep1 --output final.mp3
```

Session folder layout:

```
my-podcast-ep1/
├── 01_intro.wav
├── 02_topic_ai.wav
├── 03_mid_tune.wav
└── 04_topic_coding.wav
```

Subcommands for debugging individual stages:

```bash
podtool transcribe ./my-podcast-ep1
podtool dedup ./my-podcast-ep1 --threshold 0.9
podtool master input.wav output.mp3
```
