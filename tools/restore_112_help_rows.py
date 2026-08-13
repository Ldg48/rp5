import argparse
import csv
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import wave
import zipfile
from pathlib import Path


TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2


def part_offset(name: str) -> int:
    marker = ".part"
    if marker not in name:
        return 0
    return int(name.rsplit(marker, 1)[1] or "0")


def audio_key(value: str) -> str:
    name = str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"\.(wav|json)$", "", name, flags=re.IGNORECASE)
    matches = re.findall(r"\d+", name)
    return matches[-1] if matches else value


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


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames)
    for extra in ("source_wav_path",):
        if any(extra in row for row in rows) and extra not in fields:
            fields.append(extra)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_zip_from_tar(tar_path: Path, temp_dir: Path) -> Path:
    out_zip = temp_dir / f"{tar_path.stem}.zip"
    with tarfile.open(tar_path, "r:*") as tf, out_zip.open("wb") as out:
        members = [member for member in tf.getmembers() if member.isfile()]
        members.sort(key=lambda member: part_offset(member.name))
        for member in members:
            fh = tf.extractfile(member)
            if fh is not None:
                shutil.copyfileobj(fh, out, length=1024 * 1024 * 8)
    return out_zip


def convert_wav(src: Path, dst: Path) -> None:
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
        raise RuntimeError(result.stderr.strip() or f"ffmpeg exit {result.returncode}")
    if not is_valid_target_wav(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"converted wav failed validation: {tmp}")
    dst.unlink(missing_ok=True)
    tmp.replace(dst)


def restore_from_tar(tar_path: Path, wanted_rows: list[dict], processed_root: Path, temp_root: Path) -> int:
    wanted = {audio_key(row["audio_file_name"]): row for row in wanted_rows}
    extracted = 0
    with tempfile.TemporaryDirectory(prefix="restore_112_zip_", dir=temp_root) as tmp:
        temp_dir = Path(tmp)
        print(f"[zip] rebuilding {tar_path.name}", flush=True)
        zip_path = build_zip_from_tar(tar_path, temp_dir)
        print(f"[zip] scanning {zip_path.name}", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".wav"):
                    continue
                stem = Path(info.filename).stem
                key = audio_key(stem)
                row = wanted.get(key)
                if row is None:
                    continue
                raw_path = temp_dir / f"{key}.wav"
                with zf.open(info) as src, raw_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024 * 4)
                dst_path = processed_root / "audio" / row["split"] / row["label"] / f"{row['audio_file_name']}.wav"
                convert_wav(raw_path, dst_path)
                row["source_wav_path"] = str(raw_path)
                row["wav_path"] = str(dst_path)
                extracted += 1
                if extracted % 50 == 0:
                    print(f"[audio] restored {extracted}/{len(wanted)}", flush=True)
        zip_path.unlink(missing_ok=True)
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-manifest", type=Path, default=Path("whisper_curated_processed/manifest.csv"))
    parser.add_argument("--removed-manifest", type=Path, default=Path("whisper_curated/manifest_removed_manual.csv"))
    parser.add_argument("--processed-root", type=Path, default=Path("whisper_curated_processed"))
    parser.add_argument("--downloads", type=Path, default=Path(r"C:\Users\USER\Downloads"))
    parser.add_argument("--output", type=Path, default=Path("whisper_curated_processed/manifest_with_112.csv"))
    args = parser.parse_args()

    base_fields, base_rows = read_rows(args.processed_manifest)
    _, removed_rows = read_rows(args.removed_manifest)
    restore_rows = [row for row in removed_rows if "112" in row.get("transcript", "")]
    if not restore_rows:
        raise RuntimeError("No removed rows containing 112 were found.")

    rows_by_split: dict[str, list[dict]] = {}
    for row in restore_rows:
        rows_by_split.setdefault(row["split"], []).append(dict(row))

    tar_by_split = {
        "train": args.downloads / "1.Training 원천데이터 도움요청.tar",
        "validation": args.downloads / "2.Validation 원천데이터 도움요청.tar",
    }
    crdownload = args.downloads / "1.Training 원천데이터 도움요청.crdownload"
    if not tar_by_split["train"].exists() and crdownload.exists():
        tar_by_split["train"] = crdownload

    restored_rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="restore_112_", dir=args.processed_root) as tmp:
        temp_root = Path(tmp)
        for split, split_rows in sorted(rows_by_split.items()):
            tar_path = tar_by_split.get(split)
            if tar_path is None or not tar_path.exists():
                print(f"[skip] missing source tar for {split}: {tar_path}", flush=True)
                continue
            print(f"[restore] {split}: {len(split_rows)} rows from {tar_path}", flush=True)
            restored = restore_from_tar(tar_path, split_rows, args.processed_root, temp_root)
            print(f"[restore] {split}: restored {restored}/{len(split_rows)}", flush=True)
            restored_rows.extend([row for row in split_rows if Path(row.get("wav_path", "")).exists()])

    existing_keys = {(row.get("split"), row.get("audio_file_name")) for row in base_rows}
    new_rows = [row for row in restored_rows if (row.get("split"), row.get("audio_file_name")) not in existing_keys]
    final_rows = base_rows + new_rows
    final_rows.sort(key=lambda row: (row.get("split", ""), row.get("label", ""), row.get("audio_file_name", "")))
    write_rows(args.output, base_fields, final_rows)

    print(f"[done] base_rows={len(base_rows)} restored_rows={len(restored_rows)} added_rows={len(new_rows)}", flush=True)
    print(f"[done] output={args.output}", flush=True)
    return 0 if len(restored_rows) == len(restore_rows) else 1


if __name__ == "__main__":
    sys.exit(main())
