import argparse
import csv
import json
import random
import re
import subprocess
import wave
from collections import Counter, defaultdict
from pathlib import Path


TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2
AUDIO_EXTENSIONS = {".m4a", ".wav", ".mp3", ".aac", ".flac"}
HELP_MARKERS = (
    "119",
    "\ub3c4\uc640",
    "\uc0b4\ub824",
    "\uc2e0\uace0",
    "\uad6c\uae09\ucc28",
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def valid_wav(path: Path) -> bool:
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


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def convert_audio(src: Path, dst: Path, overwrite: bool) -> None:
    if not overwrite and valid_wav(dst):
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    tmp.unlink(missing_ok=True)
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
        tmp.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip() or f"ffmpeg failed for {src}")
    if not valid_wav(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"converted WAV failed validation: {tmp}")
    tmp.replace(dst)


def transcript_from_filename(path: Path) -> str:
    stem = path.stem.strip()
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    stem = re.sub(r"(?<!\d)\d+$", "", stem)
    return stem.strip()


def compact(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def label_for(transcript: str) -> str:
    text = compact(transcript)
    if any(marker in text for marker in HELP_MARKERS):
        return "help_direct"
    return "fall_related"


def row_key(row: dict[str, str]) -> str:
    return row.get("audio_id") or row.get("wav_path") or json.dumps(row, ensure_ascii=False, sort_keys=True)


def sample_balanced(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)

    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row.get("label", "")].append(row)

    labels = sorted(by_label)
    base_quota = limit // max(1, len(labels))
    remainder = limit % max(1, len(labels))
    selected: list[dict[str, str]] = []
    seen: set[str] = set()

    for index, label in enumerate(labels):
        group = list(by_label[label])
        rng.shuffle(group)
        quota = base_quota + (1 if index < remainder else 0)
        for row in group[:quota]:
            selected.append(row)
            seen.add(row_key(row))

    if len(selected) < limit:
        remaining = list(rows)
        rng.shuffle(remaining)
        for row in remaining:
            key = row_key(row)
            if key in seen:
                continue
            selected.append(row)
            seen.add(key)
            if len(selected) == limit:
                break

    return sorted(selected, key=lambda row: (row.get("label", ""), row.get("audio_id", "")))


def sample_balanced_with_existing(
    rows: list[dict[str, str]],
    existing_rows: list[dict[str, str]],
    target_total: int,
    seed: int,
) -> list[dict[str, str]]:
    needed = target_total - len(existing_rows)
    if needed <= 0:
        return []

    rng = random.Random(seed)
    labels = sorted({row.get("label", "") for row in rows} | {row.get("label", "") for row in existing_rows})
    existing_counts = Counter(row.get("label", "") for row in existing_rows)
    target_counts: dict[str, int] = {}
    base_quota = target_total // max(1, len(labels))
    remainder = target_total % max(1, len(labels))
    for index, label in enumerate(labels):
        target_counts[label] = base_quota + (1 if index < remainder else 0)

    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row.get("label", "")].append(row)

    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for label in labels:
        group = list(by_label[label])
        rng.shuffle(group)
        quota = max(0, target_counts[label] - existing_counts[label])
        for row in group[:quota]:
            selected.append(row)
            seen.add(row_key(row))

    if len(selected) < needed:
        remaining = list(rows)
        rng.shuffle(remaining)
        for row in remaining:
            key = row_key(row)
            if key in seen:
                continue
            selected.append(row)
            seen.add(key)
            if len(selected) == needed:
                break

    return sorted(selected[:needed], key=lambda row: (row.get("label", ""), row.get("audio_id", "")))


def used_wav_paths(manifest: Path | None) -> set[str]:
    if manifest is None or not manifest.exists():
        return set()
    _fields, rows = read_csv(manifest)
    return {str(Path(row["wav_path"]).resolve()).lower() for row in rows if row.get("wav_path")}


def build_new_recording_rows(eval_dir: Path, out_dir: Path, fieldnames: list[str], overwrite: bool) -> list[dict[str, str]]:
    files = sorted(
        [path for path in eval_dir.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS],
        key=lambda path: str(path),
    )
    rows: list[dict[str, str]] = []
    audio_dir = out_dir / "audio" / "new_recordings"
    for index, src in enumerate(files, start=1):
        transcript = transcript_from_filename(src)
        label = label_for(transcript)
        audio_id = f"new_eval_{index:04d}"
        wav_path = audio_dir / f"{audio_id}.wav"
        convert_audio(src, wav_path, overwrite=overwrite)
        duration = wav_duration(wav_path)
        row = {field: "" for field in fieldnames}
        row.update(
            {
                "split": "test",
                "label": label,
                "keyword_group": label,
                "matched_keyword": transcript,
                "category_01": "new_eval_recording",
                "category_02": transcript,
                "category_03": "held_out_user_voice",
                "transcript": transcript,
                "audio_id": audio_id,
                "audio_file_name": src.name,
                "wav_path": str(wav_path.resolve()),
                "sound_quality": "iphone",
                "annotation_start": "0",
                "annotation_end": f"{duration:.3f}",
                "audio_duration": f"{duration:.3f}",
                "sample_rate_json": str(TARGET_SAMPLE_RATE),
                "bit_rate_json": "16",
                "recording_type_json": "Mono",
                "source_tar": "",
                "source_wav_path": str(src.resolve()),
            }
        )
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, str]]) -> dict:
    return {
        "total": len(rows),
        "by_source": dict(Counter(row.get("category_01", "") for row in rows)),
        "by_label": dict(Counter(row.get("label", "") for row in rows)),
        "by_source_label": {
            f"{source}/{label}": count
            for (source, label), count in sorted(
                Counter((row.get("category_01", ""), row.get("label", "")) for row in rows).items()
            )
        },
        "new_recording_transcripts": dict(
            Counter(row.get("transcript", "") for row in rows if row.get("category_01") == "new_eval_recording")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--aihub-manifest", type=Path, default=Path("whisper_curated_processed/manifest_train8_val2_test_with_112.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("whisper_eval_mixed"))
    parser.add_argument("--aihub-test-limit", type=int, default=200)
    parser.add_argument("--target-total", type=int, default=0)
    parser.add_argument("--exclude-manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    fieldnames, aihub_rows = read_csv(args.aihub_manifest)
    new_rows = build_new_recording_rows(args.eval_dir, args.out_dir, fieldnames, args.overwrite)
    excluded_wavs = used_wav_paths(args.exclude_manifest)
    aihub_candidates = [
        dict(row, split="test")
        for row in aihub_rows
        if row.get("split") in {"validation", "test"}
        and str(Path(row.get("wav_path", "")).resolve()).lower() not in excluded_wavs
    ]
    if args.target_total > 0:
        selected_aihub = sample_balanced_with_existing(aihub_candidates, new_rows, args.target_total, args.seed)
    else:
        selected_aihub = sample_balanced(aihub_candidates, args.aihub_test_limit, args.seed)

    mixed_rows = sorted(
        selected_aihub + new_rows,
        key=lambda row: (row.get("category_01", ""), row.get("label", ""), row.get("audio_id", "")),
    )
    write_csv(args.out_dir / "manifest_new_recordings.csv", fieldnames, new_rows)
    write_csv(args.out_dir / "manifest_eval_mixed.csv", fieldnames, mixed_rows)
    summary = summarize(mixed_rows)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
