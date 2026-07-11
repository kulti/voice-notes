import argparse
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel

parser = argparse.ArgumentParser(description="Транскрибация аудио через faster-whisper.")
parser.add_argument("input_file", help="путь к аудиофайлу")
parser.add_argument("output_file", help="куда сохранить расшифровку")
parser.add_argument("--model", default="large-v3", help="размер модели (tiny/base/small/medium/large-v3/large-v3-turbo)")
parser.add_argument("--language", default="ru", help="код языка (ru, en, ...)")
parser.add_argument(
    "--prompt",
    help=(
        "initial prompt — строка со словарём имён/терминов. "
        "Whisper учитывает не более ~224 токенов с конца prompt, "
        "остальное молча игнорируется. Держи только собственные имена "
        "и редкие термины, которые действительно встречаются в аудио."
    ),
)
parser.add_argument(
    "--prompt-file",
    help="путь к файлу с initial prompt (альтернатива --prompt); действует то же ограничение ~224 токена",
)
parser.add_argument("--beam-size", type=int, default=10, help="ширина beam search")
parser.add_argument("--no-vad", action="store_true", help="отключить VAD-фильтр")
args = parser.parse_args()

initial_prompt = args.prompt
if args.prompt_file:
    if initial_prompt:
        print("Ошибка: укажи либо --prompt, либо --prompt-file, не оба.", file=sys.stderr)
        sys.exit(1)
    initial_prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

print(f"Загрузка модели {args.model}...")
model = WhisperModel(args.model, compute_type="int8")

print("Определение длительности аудио...")
segments, info = model.transcribe(
    args.input_file,
    language=args.language,
    initial_prompt=initial_prompt,
    vad_filter=not args.no_vad,
    vad_parameters=dict(min_silence_duration_ms=500),
    condition_on_previous_text=False,
    beam_size=args.beam_size,
    temperature=[0.0, 0.2, 0.4],
)

duration = info.duration
print(f"Длительность: {duration:.0f} сек. Транскрибирую...\n")

start_time = time.time()

try:
    with open(args.output_file, "w", encoding="utf-8") as f:
        for segment in segments:
            progress = segment.end / duration
            elapsed = time.time() - start_time

            if progress > 0.01:
                eta = elapsed / progress * (1 - progress)
                eta_min, eta_sec = divmod(int(eta), 60)
                eta_str = f"~{eta_min}м {eta_sec:02d}с осталось"
            else:
                eta_str = "оценка..."

            print(f"\r[{progress * 100:5.1f}%] {segment.end:.0f}/{duration:.0f} сек. | {eta_str}   ", end="", flush=True)
            f.write(segment.text.strip() + "\n")

    elapsed_total = time.time() - start_time
    mins, secs = divmod(int(elapsed_total), 60)
    print(f"\n\nГотово за {mins}м {secs:02d}с! Результат записан в {args.output_file}")

except KeyboardInterrupt:
    elapsed_total = time.time() - start_time
    mins, secs = divmod(int(elapsed_total), 60)
    print(f"\n\nПрервано пользователем после {mins}м {secs:02d}с.")
    print(f"Частичный результат сохранён в {args.output_file}")
    sys.exit(130)
