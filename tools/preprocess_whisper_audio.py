import argparse
import csv
import subprocess
import sys
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2


def is_valid_target_wav(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 44:
        return False
    try:
        with wave.open(str(path), "rb") as wav:
            return (
                wav.getframerate() == TARGET_SAMPLE_RATE
                and wav.getnchannels() == TARGET_CHANNELS
                and wav.getsampwidth() == TARGET_SAMPLE_WIDTH
                and wav.getnframes() > 0
            )
    except wave.Error:
        return False


def convert_one(row: dict, source_root: Path, out_root: Path, overwrite: bool) -> tuple[bool, dict, str]:
    src = Path(row["wav_path"])
    try:
        rel = src.resolve().relative_to(source_root.resolve())
    except ValueError:
        rel = Path(row["split"]) / row["label"] / src.name

    dst = out_root / "audio" / rel
    row = dict(row)
    row["source_wav_path"] = str(src)
    row["wav_path"] = str(dst)

    if not src.exists():
        return False, row, f"missing source: {src}"

    if not overwrite and is_valid_target_wav(dst):
        return True, row, "skipped_existing"

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp.wav")
    if tmp.exists():
        tmp.unlink()

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-sample_fmt",
        "s16",
        "-c:a",
        "pcm_s16le",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        return False, row, result.stderr.strip() or f"ffmpeg exit {result.returncode}"

    if not is_valid_target_wav(tmp):
        if tmp.exists():
            tmp.unlink()
        return False, row, "converted file failed WAV validation"

    tmp.replace(dst)
    return True, row, "converted"


def read_manifest(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    final_fields = list(fieldnames)
    if "source_wav_path" not in final_fields:
        final_fields.append("source_wav_path")
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=final_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict], converted: int, skipped: int, failed: int) -> None:
    by_split_label: dict[tuple[str, str], int] = {}
    by_category: dict[str, int] = {}
    top_transcripts: dict[str, int] = {}
    for row in rows:
        by_split_label[(row.get("split", ""), row.get("label", ""))] = (
            by_split_label.get((row.get("split", ""), row.get("label", "")), 0) + 1
        )
        by_category[row.get("category_02", "")] = by_category.get(row.get("category_02", ""), 0) + 1
        top_transcripts[row.get("transcript", "")] = top_transcripts.get(row.get("transcript", ""), 0) + 1

    lines = [
        "# Processed Whisper Dataset Summary",
        "",
        "Audio format: 16kHz mono 16-bit PCM WAV",
        "",
        f"Rows: {len(rows)}",
        f"Converted files: {converted}",
        f"Skipped existing valid files: {skipped}",
        f"Failed files: {failed}",
        "",
        "## By Split/Label",
    ]
    for (split, label), count in sorted(by_split_label.items()):
        lines.append(f"- {split} / {label}: {count}")
    lines.extend(["", "## By Category"])
    for category, count in sorted(by_category.items()):
        lines.append(f"- {category}: {count}")
    lines.extend(["", "## Top Transcripts"])
    for transcript, count in sorted(top_transcripts.items(), key=lambda item: item[1], reverse=True)[:40]:
        lines.append(f"- {transcript}: {count}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    fieldnames, rows = read_manifest(args.manifest)
    source_root = args.manifest.parent / "audio"
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    processed_rows: list[dict] = []
    failures: list[dict] = []
    converted = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(convert_one, row, source_root, out_root, args.overwrite)
            for row in rows
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            ok, row, message = future.result()
            if ok:
                processed_rows.append(row)
                if message == "skipped_existing":
                    skipped += 1
                else:
                    converted += 1
            else:
                failures.append({**row, "error": message})

            if index % 500 == 0 or index == len(rows):
                print(
                    f"[progress] {index}/{len(rows)} ok={len(processed_rows)} failed={len(failures)}",
                    flush=True,
                )

    processed_rows.sort(key=lambda row: (row.get("split", ""), row.get("label", ""), row.get("audio_id", "")))
    write_manifest(out_root / "manifest.csv", fieldnames, processed_rows)
    write_summary(out_root / "dataset_summary.md", processed_rows, converted, skipped, len(failures))

    if failures:
        failure_fields = list(fieldnames)
        for extra in ("source_wav_path", "error"):
            if extra not in failure_fields:
                failure_fields.append(extra)
        with (out_root / "preprocess_failures.csv").open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=failure_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(failures)

    print(f"[done] converted={converted} skipped={skipped} failed={len(failures)}", flush=True)
    print(f"[done] output={out_root}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
