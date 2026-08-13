import argparse
import csv
import json
import random
import shutil
import subprocess
import unicodedata
import wave
import zipfile
from collections import Counter
from pathlib import Path


TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2

HELP_MARKERS = (
    "119",
    "\ub3c4\uc640",
    "\uc0b4\ub824",
    "\uc2e0\uace0",
    "\uad6c\uae09\ucc28",
)

FIELDNAMES = [
    "split",
    "label",
    "keyword_group",
    "matched_keyword",
    "category_01",
    "category_02",
    "category_03",
    "transcript",
    "audio_id",
    "audio_file_name",
    "json_member",
    "wav_path",
    "gender",
    "generation",
    "dialect",
    "sound_quality",
    "sound_distance",
    "annotation_start",
    "annotation_end",
    "audio_duration",
    "sample_rate_json",
    "bit_rate_json",
    "recording_type_json",
    "source_tar",
    "source_wav_path",
]


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def compact(value: str) -> str:
    return nfc(value).replace(" ", "")


def label_for(transcript: str) -> str:
    text = compact(transcript)
    if any(marker in text for marker in HELP_MARKERS):
        return "help_direct"
    return "fall_related"


def is_audio_entry(entry: zipfile.ZipInfo) -> bool:
    return not entry.is_dir() and Path(entry.filename).suffix.lower() in {".m4a", ".wav", ".mp3", ".aac", ".flac"}


def transcript_from_entry(entry: zipfile.ZipInfo) -> str:
    parts = [nfc(part) for part in entry.filename.replace("\\", "/").split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"Cannot infer transcript from zip path: {entry.filename}")
    return parts[-2]


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


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stratified_split(rows: list[dict[str, str]], train_ratio: float, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    by_transcript: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_transcript.setdefault(row["transcript"], []).append(row)

    output: list[dict[str, str]] = []
    for transcript in sorted(by_transcript):
        group = sorted(by_transcript[transcript], key=lambda row: row["audio_id"])
        rng.shuffle(group)
        train_count = round(len(group) * train_ratio)
        if len(group) > 1:
            train_count = min(max(train_count, 1), len(group) - 1)
        for index, row in enumerate(group):
            copied = dict(row)
            copied["split"] = "train" if index < train_count else "test"
            output.append(copied)
    return sorted(output, key=lambda row: (row["split"], row["label"], row["transcript"], row["audio_id"]))


def summarize(rows: list[dict[str, str]]) -> dict:
    return {
        "total": len(rows),
        "by_split": dict(Counter(row["split"] for row in rows)),
        "by_label": dict(Counter(row["label"] for row in rows)),
        "by_transcript": dict(Counter(row["transcript"] for row in rows)),
        "by_split_label": {
            f"{split}/{label}": count
            for (split, label), count in sorted(Counter((row["split"], row["label"]) for row in rows).items())
        },
        "audio_format": "16kHz mono 16-bit PCM WAV",
    }


def write_summary(path: Path, all_rows: list[dict[str, str]], split_rows: list[dict[str, str]]) -> None:
    all_summary = summarize(all_rows)
    split_summary = summarize(split_rows)
    lines = [
        "# User Recordings Dataset Summary",
        "",
        "Source: iPhone recordings grouped by transcript folder.",
        "Audio format: 16kHz mono 16-bit PCM WAV",
        "",
        f"All-test rows: {all_summary['total']}",
        "",
        "## All Rows By Transcript",
    ]
    for transcript, count in sorted(all_summary["by_transcript"].items()):
        lines.append(f"- {transcript}: {count}")
    lines.extend(["", "## 8:2 Split By Split/Label"])
    for key, count in split_summary["by_split_label"].items():
        lines.append(f"- {key}: {count}")
    path.write_text("\n".join(lines), encoding="utf-8")


def import_recordings(zip_path: Path, out_dir: Path, train_ratio: float, seed: int, overwrite: bool) -> tuple[list[dict], list[dict]]:
    raw_dir = out_dir / "raw"
    audio_dir = out_dir / "audio"
    raw_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        entries = [entry for entry in zf.infolist() if is_audio_entry(entry)]
        entries.sort(key=lambda entry: nfc(entry.filename))
        for index, entry in enumerate(entries, start=1):
            transcript = transcript_from_entry(entry)
            label = label_for(transcript)
            audio_id = f"user_recording_{index:04d}"
            ext = Path(entry.filename).suffix.lower()
            raw_path = raw_dir / label / transcript / f"{audio_id}{ext}"
            wav_path = audio_dir / "all" / label / f"{audio_id}.wav"

            if overwrite or not raw_path.exists():
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(entry) as src, raw_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

            convert_audio(raw_path, wav_path, overwrite=overwrite)
            rows.append(
                {
                    "split": "test",
                    "label": label,
                    "keyword_group": label,
                    "matched_keyword": transcript,
                    "category_01": "user_recording",
                    "category_02": transcript,
                    "category_03": "mumbled_emergency_voice",
                    "transcript": transcript,
                    "audio_id": audio_id,
                    "audio_file_name": raw_path.name,
                    "json_member": "",
                    "wav_path": str(wav_path.resolve()),
                    "gender": "",
                    "generation": "",
                    "dialect": "",
                    "sound_quality": "iphone",
                    "sound_distance": "",
                    "annotation_start": "0",
                    "annotation_end": f"{wav_duration(wav_path):.3f}",
                    "audio_duration": f"{wav_duration(wav_path):.3f}",
                    "sample_rate_json": str(TARGET_SAMPLE_RATE),
                    "bit_rate_json": "16",
                    "recording_type_json": "Mono",
                    "source_tar": str(zip_path),
                    "source_wav_path": str(raw_path.resolve()),
                }
            )

    all_rows = sorted(rows, key=lambda row: (row["label"], row["transcript"], row["audio_id"]))
    split_rows = stratified_split(all_rows, train_ratio=train_ratio, seed=seed)
    return all_rows, split_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("whisper_user_recordings_processed"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    all_rows, split_rows = import_recordings(
        args.zip,
        args.out_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    write_csv(args.out_dir / "manifest_all_test.csv", all_rows)
    write_csv(args.out_dir / "manifest_train8_test2.csv", split_rows)
    write_summary(args.out_dir / "dataset_summary.md", all_rows, split_rows)

    print(json.dumps({"all_test": summarize(all_rows), "train8_test2": summarize(split_rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
