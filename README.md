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
- `--json-out path.jsonl` — additionally write a JSON Lines file with
  per-segment timings (`{start, end, text}`); required as input for
  `diarize.py apply`.

#### Prompt limits

Whisper only attends to roughly the **last ~224 tokens** of the initial
prompt; anything earlier is silently dropped. Keep the prompt tight:

- Proper nouns (character names, places, people at the meeting).
- Rare or domain-specific terms that Whisper is likely to misspell.
- Skip generic sentences — they burn tokens without improving output.
- Order matters: put the most important names near the **end** of the
  prompt, since the tail is what actually reaches the decoder.

### `diarize.py`

Adds speaker labels to a transcript ("who spoke when") using
[pyannote-audio](https://github.com/pyannote/pyannote-audio). Runs as a
two-step interactive workflow: first extract short audio samples per
detected speaker so you can identify them by ear, then merge the
mapping back into the transcript.

```bash
# 1. Transcribe with timings
python transcribe.py meeting.webm meeting.txt --json-out meeting.jsonl

# 2. Diarize + extract 3 sample clips per speaker
python diarize.py extract meeting.webm meeting.speakers/ --num-speakers 4

# 3. Listen to meeting.speakers/samples/SPEAKER_XX_N.wav and edit
#    meeting.speakers/speakers.txt, filling in real names:
#      SPEAKER_00 = Алексей
#      SPEAKER_01 = Серёжа
#      ...

# 4. Merge into a labelled Markdown file
python diarize.py apply meeting.jsonl meeting.speakers/ -o meeting.labeled.md
```

The resulting `.md` starts with a legend of `SPEAKER_XX → name` and
then every reply is prefixed with a timestamp and the speaker's name:
`` `[00:12:34]` **Алексей:** ... ``. The timestamp is the start of
the first Whisper segment in the block, ready to jump to in the
source audio/video.

Notes:
- `--num-speakers N` dramatically improves quality when you know the
  count (bounds `--min-speakers`/`--max-speakers` also work).
- Empty entries in `speakers.txt` fall through as raw `SPEAKER_XX`,
  which is fine if you want to defer naming.
- On Apple Silicon, add `--device mps` to run pyannote on the GPU
  (falls back to CPU if unstable).
- Sample selection prefers intervals where only that speaker is
  active — cleaner clips for identifying quiet participants.
- Segments are labelled by *exclusive* speaking time within them,
  not raw overlap. Quiet participants who often get talked over
  still get their lines attributed instead of losing every tie to
  the louder speaker.

#### Fixing a missing speaker (`enroll`)

If one participant (say, the one on the quietest mic) never shows up
in the transcript — pyannote failed to give them a distinct cluster
and their turns got absorbed into someone else's — you can rescue
them by hand-picking a few seconds of their clean speech and letting
the script re-label segments by voice-embedding similarity.

1. Find 1–2 intervals in the audio (5–10 s each) where **only** the
   missing person is talking.
2. Create `enrollment.txt` — one interval per line as
   `START END NAME` (seconds; name may contain spaces; `#` for
   comments):

   ```
   # Вася's clean lines
   120.5 130.0 Вася
   340 360 Вася
   ```

   You can add reference intervals for other participants too — more
   anchors, more accurate reassignment.

3. Run:

   ```bash
   python diarize.py enroll meeting.wav meeting.speakers/ enrollment.txt --device mps
   python diarize.py apply meeting.jsonl meeting.speakers/ --enrolled -o meeting.labeled.md
   ```

   `enroll` computes an average embedding per named person plus a
   centroid per existing `SPEAKER_XX` cluster (from its longest
   turns), then walks every turn ≥ 1 s and reassigns it to the
   closest candidate by cosine similarity. Shorter turns keep their
   original label (embeddings are too noisy on short clips). Output
   goes to `diarization.enroll.json`; the original file is left
   alone. `apply --enrolled` reads the enrolled version.

   Tunables: `--min-duration` (default 1.0 s) and
   `--centroid-samples` (default 8 longest turns per cluster).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Versions in `requirements.txt` are pinned. In particular, `numpy` is
held below 2.0 because the pinned `torch` (2.2.2) is compiled against
NumPy 1.x and will crash on import otherwise.

The Whisper model is downloaded on first run (default: `large-v3`;
override with `--model`). Transcription language defaults to Russian;
pass `--language en` (etc.) for other languages.

`diarize.py` additionally needs:
- `ffmpeg` in `PATH` (for cutting the sample clips).
- An `HF_TOKEN` environment variable — pyannote models are gated on
  the Hugging Face Hub. Get a token at
  <https://huggingface.co/settings/tokens>, then visit
  <https://huggingface.co/pyannote/speaker-diarization-3.1> and
  <https://huggingface.co/pyannote/segmentation-3.0> and accept the
  conditions. Export the token before running:
  `export HF_TOKEN=hf_xxx`.
