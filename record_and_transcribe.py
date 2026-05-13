import sys
import time
import threading
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from pynput import keyboard

# --- Настройки ---
CHUNK_DURATION = 60  # секунд, можно менять через аргумент
SAMPLE_RATE = 16000
OUTPUT_FILE = "заметка.txt"

if len(sys.argv) >= 2:
    OUTPUT_FILE = sys.argv[1]
if len(sys.argv) >= 3:
    CHUNK_DURATION = int(sys.argv[2])

# --- Состояние ---
recording = False
finished = False
audio_chunks = queue.Queue()
current_audio = []
lock = threading.Lock()

def on_press(key):
    global recording, finished
    try:
        if key == keyboard.Key.space:
            with lock:
                if not finished:
                    recording = not recording
                    state = "⏺  Запись..." if recording else "⏸  Пауза"
                    print(f"\r{state}                    ", end="", flush=True)
        elif key == keyboard.Key.enter:
            with lock:
                recording = False
                finished = True
            print(f"\r⏹  Завершение записи...                    ", flush=True)
            return False  # остановить listener
    except AttributeError:
        pass

def audio_callback(indata, frames, time_info, status):
    if recording:
        with lock:
            current_audio.append(indata.copy())

def flush_chunk():
    """Сбрасывает накопленное аудио в очередь как чанк."""
    with lock:
        if current_audio:
            chunk = np.concatenate(current_audio, axis=0)
            current_audio.clear()
            audio_chunks.put(chunk)

def recording_thread():
    """Поток, который нарезает аудио на чанки по таймеру."""
    while not finished:
        time.sleep(CHUNK_DURATION)
        if not finished:
            flush_chunk()
            count = audio_chunks.qsize()
            if count > 0:
                print(f"\r⏺  Запись... (чанков в очереди: {count})", end="", flush=True)

def transcribe_thread(model):
    """Поток, который транскрибирует чанки по мере их появления."""
    chunk_index = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        while True:
            try:
                chunk = audio_chunks.get(timeout=1)
            except queue.Empty:
                if finished and audio_chunks.empty():
                    break
                continue

            chunk_index += 1
            duration = len(chunk) / SAMPLE_RATE
            print(f"\r🔄 Транскрибирую чанк {chunk_index} ({duration:.0f} сек.)...          ", end="", flush=True)

            # Сохраняем во временный файл
            tmp_path = Path(f"/tmp/chunk_{chunk_index}.wav")
            sf.write(tmp_path, chunk, SAMPLE_RATE)

            segments, _ = model.transcribe(str(tmp_path), language="ru")
            for segment in segments:
                f.write(segment.text.strip() + "\n")
            f.flush()

            tmp_path.unlink()
            print(f"\r✅ Чанк {chunk_index} готов.                              ", end="", flush=True)

    print()

def main():
    print("Загрузка модели...")
    from faster_whisper import WhisperModel
    model = WhisperModel("medium", compute_type="int8")

    print(f"\nНастройки: чанк = {CHUNK_DURATION} сек., выход = {OUTPUT_FILE}")
    print("─" * 50)
    print("Пробел  — начать/пауза")
    print("Enter   — завершить запись")
    print("Ctrl+C  — прервать")
    print("─" * 50)
    print("⏸  Пауза. Нажмите пробел для начала записи.")

    # Запускаем потоки
    t_transcribe = threading.Thread(target=transcribe_thread, args=(model,), daemon=True)
    t_transcribe.start()

    t_chunker = threading.Thread(target=recording_thread, daemon=True)
    t_chunker.start()

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
            with keyboard.Listener(on_press=on_press) as listener:
                listener.join()

        # Запись завершена — сбрасываем остаток
        flush_chunk()

        print("⏳ Дотранскрибирую оставшееся...")
        t_transcribe.join()

        print(f"\n✅ Готово! Результат в {OUTPUT_FILE}")

    except KeyboardInterrupt:
        print(f"\n\n⛔ Прервано. Частичный результат может быть в {OUTPUT_FILE}")
        sys.exit(130)

if __name__ == "__main__":
    main()
