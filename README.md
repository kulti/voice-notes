# voice-notes

Simple utilities for transcribing voice notes (Russian by default) using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Scripts

### `record_and_transcribe.py`

Records from the microphone and transcribes in parallel, splitting audio
into chunks so the transcript grows while you keep talking.

```bash
python record_and_transcribe.py [output_file] [chunk_duration_sec]
```

Defaults: `заметка.txt`, 60-second chunks.

Controls:
- `Space` — start / pause
- `Enter` — finish recording
- `Ctrl+C` — abort

### `transcribe.py`

Transcribes an existing audio file, printing progress and ETA.

```bash
python transcribe.py <audio_file> <output_file> [options]
```

Useful options:
- `--model large-v3` (default) — model size; try `large-v3-turbo` for a
  faster large-quality run, or `medium` if you're memory-constrained.
- `--language ru` (default) — target language.
- `--prompt "..."` / `--prompt-file path.txt` — initial prompt biasing
  Whisper toward specific proper nouns and rare terms.
- `--beam-size 10` (default) — larger = more accurate, slower.
- `--no-vad` — disable the voice-activity filter (on by default; it
  suppresses hallucinated text in silent regions).

#### Prompt limits

Whisper only attends to roughly the **last ~224 tokens** of the initial
prompt; anything earlier is silently dropped. Keep the prompt tight:

- Proper nouns (character names, places, people at the meeting).
- Rare or domain-specific terms that Whisper is likely to misspell.
- Skip generic sentences — they burn tokens without improving output.
- Order matters: put the most important names near the **end** of the
  prompt, since the tail is what actually reaches the decoder.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install faster-whisper sounddevice soundfile numpy pynput
```

The Whisper model is downloaded on first run (default: `large-v3`;
override with `--model`). Transcription language defaults to Russian;
pass `--language en` (etc.) for other languages.
