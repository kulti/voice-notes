"""Диаризация: разметка «кто говорит когда» через pyannote и слияние с транскриптом.

Двухшаговый рабочий процесс:

1) extract — прогоняет pyannote по аудио, кластеризует спикеров, извлекает
   3 сэмпла на каждого (в samples/) и создаёт шаблон speakers.txt.
   Ты слушаешь сэмплы, вписываешь имена справа от `=`.

2) apply — берёт speakers.txt + JSONL-транскрипт (от transcribe.py --json-out)
   и выдаёт итоговый .md, где каждая реплика подписана именем спикера.
   Легенда (SPEAKER_XX → имя) добавляется в начало файла.

3) enroll (опционально) — если pyannote не выделил кого-то в отдельный
   кластер (тихий микрофон, частые перебивания), ты вручную указываешь
   1–2 интервала чистой речи этого участника в enrollment.txt, а команда
   считает его голосовой эмбеддинг и переприсваивает метки сегментов по
   косинусной близости. Пишет diarization.enroll.json, который apply
   подхватит с флагом --enrolled.

Требуется:
  pip install pyannote.audio soundfile
  export HF_TOKEN=hf_xxx       # https://huggingface.co/settings/tokens
  плюс принять условия моделей:
    https://huggingface.co/pyannote/speaker-diarization-3.1
    https://huggingface.co/pyannote/segmentation-3.0
  ffmpeg в PATH (нужен для нарезки сэмплов).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def require_wav(src: str) -> None:
    """pyannote читает через libsndfile, который не понимает webm/mp4/m4a.
    Подскажем пользователю команду ffmpeg — WAV сохранится и переживёт перезапуски."""
    if src.lower().endswith(".wav"):
        return
    wav = str(Path(src).with_suffix(".wav"))
    sys.exit(
        f"pyannote не читает {Path(src).suffix} напрямую. Сконвертируй в WAV:\n"
        f"  ffmpeg -i {src!r} -ac 1 -ar 16000 {wav!r}\n"
        f"Затем запусти diarize.py с этим .wav."
    )


def cmd_extract(args):
    require_wav(args.audio)
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        sys.exit(
            "Нужен pyannote.audio. Установи:\n"
            "  pip install pyannote.audio soundfile"
        )

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit(
            "HF_TOKEN не задан. Токен нужен, чтобы скачать модели pyannote.\n"
            "  1) Получи токен: https://huggingface.co/settings/tokens\n"
            "  2) Прими условия обеих моделей:\n"
            "       https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "       https://huggingface.co/pyannote/segmentation-3.0\n"
            "  3) export HF_TOKEN=hf_xxx"
        )

    print("Загрузка пайплайна pyannote/speaker-diarization-3.1...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    if args.device != "cpu":
        try:
            import torch

            pipeline.to(torch.device(args.device))
            print(f"Устройство: {args.device}")
        except Exception as e:
            print(f"Не смог переключиться на {args.device}, остаюсь на CPU: {e}", file=sys.stderr)

    kwargs = {}
    if args.num_speakers:
        kwargs["num_speakers"] = args.num_speakers
    else:
        if args.min_speakers:
            kwargs["min_speakers"] = args.min_speakers
        if args.max_speakers:
            kwargs["max_speakers"] = args.max_speakers

    print("Запуск диаризации (может идти минуты)...")
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    with ProgressHook() as hook:
        diarization = pipeline(args.audio, hook=hook, **kwargs)

    turns_by_speaker: dict[str, list[tuple[float, float]]] = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns_by_speaker.setdefault(speaker, []).append((turn.start, turn.end))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    all_turns = [
        {"speaker": spk, "start": s, "end": e}
        for spk, ts in turns_by_speaker.items()
        for s, e in ts
    ]
    all_turns.sort(key=lambda t: t["start"])
    (out_dir / "diarization.json").write_text(
        json.dumps({"turns": all_turns}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Нарезаю сэмплы в {samples_dir}...")
    for speaker in sorted(turns_by_speaker):
        own = turns_by_speaker[speaker]
        others = [
            interval
            for other_spk, ts in turns_by_speaker.items()
            if other_spk != speaker
            for interval in ts
        ]
        # Берём куски, где говорит ТОЛЬКО этот спикер — иначе для «тихих» участников
        # самые длинные turns окажутся перекрытиями и сэмпл будет неразборчивым.
        clean = subtract_intervals(own, others)
        candidates = clean or own
        candidates.sort(key=lambda t: t[1] - t[0], reverse=True)
        picked = candidates[: args.samples_per_speaker]
        for i, (start, end) in enumerate(picked, 1):
            start = start + 0.3
            end = min(end - 0.3, start + args.sample_seconds)
            if end <= start:
                continue
            sample_path = samples_dir / f"{speaker}_{i}.wav"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
                    "-i", args.audio,
                    "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
                    "-ac", "1", "-ar", "16000",
                    str(sample_path),
                ],
                check=True,
            )

    speakers_txt = out_dir / "speakers.txt"
    if not speakers_txt.exists() or args.overwrite_speakers:
        lines = [
            "# Прослушай сэмплы в samples/ и впиши имя каждого спикера справа от =",
            "# Пример: SPEAKER_00 = Алексей",
            "# Пустое имя = оставить SPEAKER_XX в итоговом файле",
            "",
        ]
        for speaker in sorted(turns_by_speaker):
            lines.append(f"{speaker} = ")
        speakers_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"\nОбнаружено спикеров: {len(turns_by_speaker)}\n"
        f"Сэмплы: {samples_dir}\n"
        f"Разметь: {speakers_txt}\n"
        f"Дальше:  python diarize.py apply <транскрипт.jsonl> {out_dir} -o labeled.md"
    )


def read_speakers(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def subtract_intervals(
    base: list[tuple[float, float]],
    others: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Возвращает интервалы из base с вырезанным объединением others."""
    if not others:
        return list(base)
    sorted_others = sorted(others)
    merged: list[list[float]] = []
    for s, e in sorted_others:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    result: list[tuple[float, float]] = []
    for s, e in base:
        cur = s
        for os_, oe in merged:
            if oe <= cur:
                continue
            if os_ >= e:
                break
            if os_ > cur:
                result.append((cur, os_))
            cur = max(cur, oe)
            if cur >= e:
                break
        if cur < e:
            result.append((cur, e))
    return result


def dominant_speaker(start: float, end: float, turns: list[dict]) -> str | None:
    by_speaker: dict[str, list[tuple[float, float]]] = {}
    for t in turns:
        s = max(start, t["start"])
        e = min(end, t["end"])
        if e > s:
            by_speaker.setdefault(t["speaker"], []).append((s, e))
    if not by_speaker:
        return None

    # Считаем время, где спикер говорит ЭКСКЛЮЗИВНО (без наложения на других).
    # Иначе тихий участник, чья речь почти всегда пересекается с чужой, никогда
    # не выиграет по суммарному перекрытию и его реплики выпадут из результата.
    exclusive: dict[str, float] = {}
    for spk, intervals in by_speaker.items():
        others = [i for other_spk, ints in by_speaker.items() if other_spk != spk for i in ints]
        clean = subtract_intervals(intervals, others)
        exclusive[spk] = sum(e - s for s, e in clean)

    if any(v > 0 for v in exclusive.values()):
        return max(exclusive.items(), key=lambda kv: kv[1])[0]

    # Фоллбэк: сегмент целиком в зоне наложения — возвращаемся к суммарному пересечению.
    totals = {spk: sum(e - s for s, e in ints) for spk, ints in by_speaker.items()}
    return max(totals.items(), key=lambda kv: kv[1])[0]


def read_enrollment(path: Path) -> dict[str, list[tuple[float, float]]]:
    """Формат строки: `START END NAME` (имя может содержать пробелы). Комментарии — `#`."""
    result: dict[str, list[tuple[float, float]]] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            sys.exit(f"{path}:{lineno}: ожидаю `START END NAME`, получил: {raw!r}")
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError:
            sys.exit(f"{path}:{lineno}: START/END должны быть числами, получил: {raw!r}")
        if end <= start:
            sys.exit(f"{path}:{lineno}: END должен быть больше START")
        name = parts[2].strip()
        result.setdefault(name, []).append((start, end))
    return result


def cmd_enroll(args):
    require_wav(args.audio)
    try:
        from pyannote.audio import Inference, Model
        from pyannote.core import Segment
        import numpy as np
    except ImportError:
        sys.exit("Нужны pyannote.audio и numpy. pip install -r requirements.txt")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("HF_TOKEN не задан — модель эмбеддинга скачивается с Hugging Face.")

    speakers_dir = Path(args.speakers_dir)
    enrollment_path = Path(args.enrollment)
    enrollments = read_enrollment(enrollment_path)
    if not enrollments:
        sys.exit(f"В {enrollment_path} нет ни одного интервала.")

    diarization_path = speakers_dir / "diarization.json"
    turns = json.loads(diarization_path.read_text(encoding="utf-8"))["turns"]

    print("Загрузка модели эмбеддингов pyannote/wespeaker-voxceleb-resnet34-LM...")
    model = Model.from_pretrained(
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        use_auth_token=hf_token,
    )
    if args.device != "cpu":
        try:
            import torch
            model.to(torch.device(args.device))
        except Exception as e:
            print(f"Не смог переключиться на {args.device}, остаюсь на CPU: {e}", file=sys.stderr)
    inference = Inference(model, window="whole")

    def embed(start: float, end: float):
        emb = inference.crop(args.audio, Segment(start, end))
        arr = np.asarray(emb)
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        norm = np.linalg.norm(arr)
        return arr / norm if norm > 0 else arr

    # Эталонные эмбеддинги: среднее по указанным пользователем интервалам.
    print(f"Считаю эталонные эмбеддинги для: {', '.join(enrollments)}")
    reference: dict[str, "np.ndarray"] = {}
    for name, intervals in enrollments.items():
        embs = [embed(s, e) for s, e in intervals]
        avg = np.mean(embs, axis=0)
        n = np.linalg.norm(avg)
        reference[name] = avg / n if n > 0 else avg

    # Центроиды существующих кластеров: усредняем по нескольким самым длинным turns
    # (короткие turns дают шумный эмбеддинг, к тому же считать все — долго).
    by_speaker: dict[str, list[tuple[float, float]]] = {}
    for t in turns:
        by_speaker.setdefault(t["speaker"], []).append((t["start"], t["end"]))
    print(f"Считаю центроиды кластеров ({len(by_speaker)} шт., по {args.centroid_samples} самых длинных turns)...")
    centroids: dict[str, "np.ndarray"] = {}
    for spk, ints in by_speaker.items():
        longest = sorted(ints, key=lambda t: t[1] - t[0], reverse=True)[: args.centroid_samples]
        longest = [(s, e) for s, e in longest if e - s >= args.min_duration]
        if not longest:
            continue
        embs = [embed(s, e) for s, e in longest]
        avg = np.mean(embs, axis=0)
        n = np.linalg.norm(avg)
        centroids[spk] = avg / n if n > 0 else avg

    candidates = {**centroids, **reference}  # эталоны перекрывают одноимённые кластеры

    # Проходим по всем turns: длинные — переприсваиваем по близости эмбеддинга,
    # короткие оставляем со старой меткой (эмбеддинг слишком шумный).
    long_turns = [t for t in turns if t["end"] - t["start"] >= args.min_duration]
    print(f"Переприсваиваю метки {len(long_turns)}/{len(turns)} сегментов (короче {args.min_duration}с — не трогаю)...")
    reassigned = 0
    for i, t in enumerate(turns):
        if t["end"] - t["start"] < args.min_duration:
            continue
        emb = embed(t["start"], t["end"])
        best_name, best_sim = None, -1.0
        for name, ref in candidates.items():
            sim = float(np.dot(emb, ref))
            if sim > best_sim:
                best_name, best_sim = name, sim
        if best_name and best_name != t["speaker"]:
            t["speaker"] = best_name
            reassigned += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(turns)}", end="\r", flush=True)
    print(f"\nПереприсвоено меток: {reassigned}")

    out_path = speakers_dir / "diarization.enroll.json"
    out_path.write_text(
        json.dumps({"turns": turns}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Дописываем в speakers.txt строки для новых имён, чтобы легенда была осмысленной.
    speakers_txt = speakers_dir / "speakers.txt"
    mapping = read_speakers(speakers_txt)
    new_names = [n for n in reference if n not in mapping]
    if new_names:
        with speakers_txt.open("a", encoding="utf-8") as f:
            f.write("\n# Добавлено enroll:\n")
            for name in new_names:
                f.write(f"{name} = {name}\n")

    print(
        f"\nГотово: {out_path}\n"
        f"Дальше:  python diarize.py apply <транскрипт.jsonl> {speakers_dir} --enrolled -o labeled.md"
    )


def cmd_apply(args):
    speakers_dir = Path(args.speakers_dir)
    mapping = read_speakers(speakers_dir / "speakers.txt")
    diar_name = "diarization.enroll.json" if args.enrolled else "diarization.json"
    turns = json.loads((speakers_dir / diar_name).read_text(encoding="utf-8"))["turns"]

    segments = []
    with open(args.transcript, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                segments.append(json.loads(line))

    def label_for(speaker_id: str | None) -> str:
        if not speaker_id:
            return "?"
        return mapping.get(speaker_id) or speaker_id

    all_speakers = sorted({t["speaker"] for t in turns})
    legend = ["# Легенда спикеров", ""]
    for spk in all_speakers:
        legend.append(f"- **{spk}** — {mapping.get(spk) or '(не размечен)'}")
    legend.append("")

    def fmt_time(seconds: float) -> str:
        total = int(seconds)
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    body: list[str] = []
    current: str | None = None
    block_start: float = 0.0
    buffer: list[str] = []
    for seg in segments:
        name = label_for(dominant_speaker(seg["start"], seg["end"], turns))
        if name != current:
            if buffer:
                body.append(f"`[{fmt_time(block_start)}]` **{current}:** {' '.join(buffer)}")
                buffer = []
            current = name
            block_start = seg["start"]
        buffer.append(seg["text"].strip())
    if buffer:
        body.append(f"`[{fmt_time(block_start)}]` **{current}:** {' '.join(buffer)}")

    output = "\n".join(legend) + "\n" + "\n\n".join(body) + "\n"

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Готово: {args.output}")
    else:
        sys.stdout.write(output)


def main():
    parser = argparse.ArgumentParser(
        description="Диаризация: разметка спикеров через pyannote и слияние с транскриптом.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    ext = sub.add_parser("extract", help="запустить диаризацию и извлечь аудио-сэмплы на каждого спикера")
    ext.add_argument("audio", help="аудиофайл (webm/mp4/wav/m4a — всё, что читает ffmpeg)")
    ext.add_argument("output_dir", help="куда положить diarization.json, speakers.txt и samples/")
    ext.add_argument("--num-speakers", type=int, help="точное количество спикеров (сильно улучшает качество)")
    ext.add_argument("--min-speakers", type=int, help="минимум, если точное число неизвестно")
    ext.add_argument("--max-speakers", type=int, help="максимум, если точное число неизвестно")
    ext.add_argument("--samples-per-speaker", type=int, default=3, help="сколько сэмплов извлечь на спикера")
    ext.add_argument("--sample-seconds", type=float, default=10.0, help="длина одного сэмпла в секундах")
    ext.add_argument("--device", default="cpu", help="cpu / mps / cuda")
    ext.add_argument("--overwrite-speakers", action="store_true", help="переписать speakers.txt даже если уже существует")
    ext.set_defaults(func=cmd_extract)

    app = sub.add_parser("apply", help="слить размеченный speakers.txt с JSONL-транскриптом в итоговый .md")
    app.add_argument("transcript", help="файл .jsonl от transcribe.py --json-out")
    app.add_argument("speakers_dir", help="папка от команды extract")
    app.add_argument("-o", "--output", help="куда сохранить .md (по умолчанию — stdout)")
    app.add_argument(
        "--enrolled",
        action="store_true",
        help="использовать diarization.enroll.json (после команды enroll) вместо оригинального",
    )
    app.set_defaults(func=cmd_apply)

    enr = sub.add_parser("enroll", help="переприсвоить метки сегментов по эталонным эмбеддингам голосов")
    enr.add_argument("audio", help="то же аудио, что и в extract")
    enr.add_argument("speakers_dir", help="папка от команды extract")
    enr.add_argument(
        "enrollment",
        help=(
            "текстовый файл с интервалами эталонной речи. Формат строки: "
            "`START END NAME` (секунды, имя может содержать пробелы). "
            "Комментарии — `#`. Можно несколько строк на одного человека."
        ),
    )
    enr.add_argument("--device", default="cpu", help="cpu / mps / cuda")
    enr.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="turns короче этого не переприсваиваются (эмбеддинг слишком шумный)",
    )
    enr.add_argument(
        "--centroid-samples",
        type=int,
        default=8,
        help="сколько самых длинных turns кластера усреднять в его центроид",
    )
    enr.set_defaults(func=cmd_enroll)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
