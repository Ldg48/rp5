import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

KEYWORDS = [
    "도와주세요",
    "살려주세요",
    "사람살려",
    "119",
    "구급차",
    "신고",
    "넘어졌",
    "미끄러졌",
    "못일어나",
    "일어날수없",
    "다쳤",
    "아파",
    "허리아파",
    "다리아파",
    "머리아파",
    "아이고",
]


def normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w가-힣0-9]", "", text)
    return text


def levenshtein(a: list[str] | str, b: list[str] | str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = cur
    return prev[-1]


def cer(reference: str, prediction: str) -> float:
    ref = normalize(reference)
    pred = normalize(prediction)
    if not ref:
        return 0.0 if not pred else 1.0
    return levenshtein(ref, pred) / len(ref)


def wer(reference: str, prediction: str) -> float:
    ref = (reference or "").strip().split()
    pred = (prediction or "").strip().split()
    if not ref:
        return 0.0 if not pred else 1.0
    return levenshtein(ref, pred) / len(ref)


def keyword_hit(reference: str, prediction: str) -> bool:
    ref = normalize(reference)
    pred = normalize(prediction)
    targets = [kw for kw in KEYWORDS if kw in ref]
    if not targets:
        return True
    return any(kw in pred for kw in targets)


def load_manifest(path: Path, split: str, limit: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = [row for row in csv.DictReader(fh) if row.get("split") == split]
    rows.sort(key=lambda row: (row.get("label", ""), row.get("audio_id", "")))
    if limit:
        by_label: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_label.setdefault(row.get("label", ""), []).append(row)
        labels = sorted(by_label)
        selected = []
        per_label = max(1, limit // max(1, len(labels)))
        for label in labels:
            selected.extend(by_label[label][:per_label])
        rows = selected[:limit]
    return rows


def build_generate_kwargs(language: str, max_new_tokens: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if language:
        kwargs.update({"language": language, "task": "transcribe"})
    if max_new_tokens > 0:
        kwargs["max_new_tokens"] = max_new_tokens
    kwargs.update(
        {
            "num_beams": 1,
            "do_sample": False,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
        }
    )
    return kwargs


def build_transformers_pipeline(model_id: str, adapter: str, device: int, dtype: Any) -> Any:
    from transformers import pipeline

    if not adapter:
        return pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype=dtype,
        )

    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor_path = adapter if Path(adapter).exists() else model_id
    processor = WhisperProcessor.from_pretrained(processor_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        torch_dtype=dtype,
    )


def transcribe_with_transformers(
    model_id: str,
    adapter: str,
    rows: list[dict[str, str]],
    language: str,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    import torch

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = build_transformers_pipeline(model_id, adapter, device, dtype)
    generate_kwargs = build_generate_kwargs(language, max_new_tokens)

    results = []
    for index, row in enumerate(rows, start=1):
        started = time.time()
        try:
            if generate_kwargs:
                output = pipe(row["wav_path"], generate_kwargs=generate_kwargs)
            else:
                output = pipe(row["wav_path"])
        except ValueError as exc:
            if "generation config is outdated" not in str(exc):
                raise
            output = pipe(row["wav_path"])
        pred = (output.get("text") or "").strip()
        elapsed = time.time() - started
        results.append({**row, "prediction": pred, "seconds": f"{elapsed:.4f}"})
        if index % 20 == 0 or index == len(rows):
            print(f"[progress] {index}/{len(rows)}", flush=True)
    return results


def transcribe_with_openai_whisper(model_name: str, rows: list[dict[str, str]], language: str) -> list[dict[str, str]]:
    import whisper

    model = whisper.load_model(model_name)
    results = []
    for index, row in enumerate(rows, start=1):
        started = time.time()
        output = model.transcribe(row["wav_path"], language=language, fp16=False)
        pred = (output.get("text") or "").strip()
        elapsed = time.time() - started
        results.append({**row, "prediction": pred, "seconds": f"{elapsed:.4f}"})
        if index % 20 == 0 or index == len(rows):
            print(f"[progress] {index}/{len(rows)}", flush=True)
    return results


def transcribe_with_faster_whisper(
    model_path: str,
    rows: list[dict[str, str]],
    language: str,
    device: str,
    compute_type: str,
    beam_size: int,
    cpu_threads: int,
    num_workers: int,
) -> list[dict[str, str]]:
    from faster_whisper import WhisperModel

    model_kwargs: dict[str, Any] = {"device": device, "compute_type": compute_type}
    if cpu_threads > 0:
        model_kwargs["cpu_threads"] = cpu_threads
    if num_workers > 0:
        model_kwargs["num_workers"] = num_workers
    model = WhisperModel(model_path, **model_kwargs)
    results = []
    transcribe_language = language or None
    for index, row in enumerate(rows, start=1):
        started = time.time()
        segments, _info = model.transcribe(
            row["wav_path"],
            language=transcribe_language,
            task="transcribe",
            beam_size=beam_size,
            without_timestamps=True,
        )
        pred = "".join(segment.text for segment in segments).strip()
        elapsed = time.time() - started
        results.append({**row, "prediction": pred, "seconds": f"{elapsed:.4f}"})
        if index % 20 == 0 or index == len(rows):
            print(f"[progress] {index}/{len(rows)}", flush=True)
    return results


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    extra = ["cer", "wer", "keyword_hit"]
    for name in extra:
        if name not in fieldnames:
            fieldnames.append(name)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)
    avg_cer = sum(float(row["cer"]) for row in rows) / total if total else 0.0
    avg_wer = sum(float(row["wer"]) for row in rows) / total if total else 0.0
    hit_rate = sum(1 for row in rows if row["keyword_hit"] == "true") / total if total else 0.0
    by_label = {}
    labels = sorted(set(row["label"] for row in rows))
    for label in labels:
        subset = [row for row in rows if row["label"] == label]
        by_label[label] = {
            "count": len(subset),
            "cer": sum(float(row["cer"]) for row in subset) / len(subset),
            "wer": sum(float(row["wer"]) for row in subset) / len(subset),
            "keyword_hit_rate": sum(1 for row in subset if row["keyword_hit"] == "true") / len(subset),
        }
    return {
        "count": total,
        "cer": avg_cer,
        "wer": avg_wer,
        "keyword_hit_rate": hit_rate,
        "by_label": by_label,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--backend", choices=["transformers", "openai-whisper", "faster-whisper"], default="transformers")
    parser.add_argument("--model", default="seastar105/whisper-small-ko-zeroth")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--language", default="")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    if args.shard_count < 1:
        raise SystemExit("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit("--shard-index must satisfy 0 <= index < shard-count")

    rows = load_manifest(args.manifest, args.split, args.limit or None)
    if args.shard_count > 1:
        rows = [row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index]
    if not rows:
        raise SystemExit(f"No rows found for split={args.split}")

    print(
        f"[baseline] rows={len(rows)} model={args.model} adapter={args.adapter or '-'} "
        f"backend={args.backend} shard={args.shard_index}/{args.shard_count}",
        flush=True,
    )
    if args.backend == "transformers":
        results = transcribe_with_transformers(
            args.model,
            args.adapter,
            rows,
            args.language,
            args.max_new_tokens,
        )
    elif args.backend == "faster-whisper":
        results = transcribe_with_faster_whisper(
            args.model,
            rows,
            args.language,
            args.device,
            args.compute_type,
            args.beam_size,
            args.cpu_threads,
            args.num_workers,
        )
    else:
        results = transcribe_with_openai_whisper(args.model, rows, args.language)

    scored = []
    for row in results:
        ref = row.get("transcript", "")
        pred = row.get("prediction", "")
        scored.append(
            {
                **row,
                "cer": f"{cer(ref, pred):.6f}",
                "wer": f"{wer(ref, pred):.6f}",
                "keyword_hit": "true" if keyword_hit(ref, pred) else "false",
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "baseline_predictions.csv", scored)
    summary = summarize(scored)
    (args.out_dir / "baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
