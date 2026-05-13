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
python transcribe.py <audio_file> <output_file>
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install faster-whisper sounddevice soundfile numpy pynput
```

The `medium` Whisper model is downloaded on first run. Transcription
language is hard-coded to Russian (`language="ru"`) — change it in the
source if you need a different language.
