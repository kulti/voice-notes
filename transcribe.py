import sys
import time
from faster_whisper import WhisperModel

if len(sys.argv) < 3:
    print("Использование: python transcribe.py <аудио_файл> <выходной_файл>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

print("Загрузка модели...")
model = WhisperModel("medium", compute_type="int8")

print("Определение длительности аудио...")
segments, info = model.transcribe(input_file, language="ru")

duration = info.duration
print(f"Длительность: {duration:.0f} сек. Транскрибирую...\n")

start_time = time.time()

try:
    with open(output_file, "w", encoding="utf-8") as f:
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
    print(f"\n\nГотово за {mins}м {secs:02d}с! Результат записан в {output_file}")

except KeyboardInterrupt:
    elapsed_total = time.time() - start_time
    mins, secs = divmod(int(elapsed_total), 60)
    print(f"\n\nПрервано пользователем после {mins}м {secs:02d}с.")
    print(f"Частичный результат сохранён в {output_file}")
    sys.exit(130)
