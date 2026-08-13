import argparse
import csv
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


KEYWORDS = {
    "help_direct": [
        "도와주세요",
        "살려주세요",
        "사람 살려",
        "119 불러줘",
        "구급차 불러줘",
        "신고해주세요",
    ],
    "fall_related": [
        "넘어졌어요",
        "미끄러졌어요",
        "못 일어나겠어요",
        "일어날 수 없어요",
        "다쳤어요",
    ],
    "pain": [
        "아파요",
        "허리 아파요",
        "다리 아파요",
        "머리 아파요",
        "아이고",
    ],
}


FILES = {
    ("train", "fall", "label"): "1.Training 낙상.tar",
    ("train", "help", "label"): "1.Training 도움요청.tar",
    ("train", "fall", "audio"): "1.Training 원천데이터 낙상.tar",
    ("train", "help", "audio"): "1.Training 원천데이터 도움요청.tar",
    ("validation", "fall", "label"): "2.Validation 낙상.tar",
    ("validation", "help", "label"): "2.Validation 도움요청.tar",
    ("validation", "fall", "audio"): "2.Validation 원천데이터 낙상.tar",
    ("validation", "help", "audio"): "2.Validation 원천데이터 도움요청.tar",
}


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
]


def norm_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


def match_keyword(note: str):
    normalized_note = norm_text(note)
    for group, words in KEYWORDS.items():
        for word in words:
            if norm_text(word) in normalized_note:
                return group, word
    return "", ""


def decode_zip_name(name: str) -> str:
    try:
        return name.encode("cp437").decode("cp949")
    except Exception:
        return name


def part_offset(name: str) -> int:
    marker = ".part"
    if marker not in name:
        return 0
    tail = name.rsplit(marker, 1)[1]
    return int(tail or "0")


def read_single_part_zip_from_tar(tar_path: Path) -> bytes:
    with tarfile.open(tar_path, "r:*") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        if not members:
            raise RuntimeError(f"No file entries in {tar_path}")
        members.sort(key=lambda m: part_offset(m.name))
        chunks = []
        for member in members:
            fh = tf.extractfile(member)
            if fh is None:
                continue
            chunks.append(fh.read())
        return b"".join(chunks)


def iter_label_rows(raw_dir: Path):
    for (split, category, kind), file_name in FILES.items():
        if kind != "label":
            continue
        tar_path = raw_dir / file_name
        zip_bytes = read_single_part_zip_from_tar(tar_path)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for info in zf.infolist():
                decoded = decode_zip_name(info.filename)
                if info.is_dir() or not decoded.lower().endswith(".json"):
                    continue
                with zf.open(info) as fh:
                    obj = json.loads(fh.read().decode("utf-8-sig"))
                annotations = obj.get("annotations") or []
                if not annotations:
                    continue
                ann = annotations[0]
                cats = ann.get("categories") or {}
                note = ann.get("note") or ""
                keyword_group, matched_keyword = match_keyword(note)
                category_02 = cats.get("category_02", "")
                include = False
                label = ""
                if category_02 == "낙상":
                    include = True
                    label = "fall_related"
                    if keyword_group:
                        label = keyword_group
                elif category_02 == "도움요청":
                    include = bool(keyword_group)
                    label = keyword_group or "excluded_unmatched_help"
                else:
                    include = bool(keyword_group)
                    label = keyword_group or "excluded_other"

                audio = obj.get("audio") or {}
                area = ann.get("area") or {}
                row = {
                    "split": split,
                    "label": label,
                    "keyword_group": keyword_group,
                    "matched_keyword": matched_keyword,
                    "category_01": cats.get("category_01", ""),
                    "category_02": category_02,
                    "category_03": cats.get("category_03", ""),
                    "transcript": note,
                    "audio_id": ann.get("audio_id", ""),
                    "audio_file_name": audio.get("fileName", ""),
                    "json_member": decoded,
                    "wav_path": "",
                    "gender": ann.get("gender", ""),
                    "generation": ann.get("generation", ""),
                    "dialect": ann.get("dialect", ""),
                    "sound_quality": ann.get("soundQuality", ""),
                    "sound_distance": ann.get("soundDistance", ""),
                    "annotation_start": area.get("start", ""),
                    "annotation_end": area.get("end", ""),
                    "audio_duration": audio.get("duration", ""),
                    "sample_rate_json": audio.get("sampleRate", ""),
                    "bit_rate_json": audio.get("bitRate", ""),
                    "recording_type_json": audio.get("recodingType", ""),
                    "source_tar": file_name,
                    "_include": include,
                    "_source_category": category,
                }
                yield row


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def write_keywords(path: Path):
    path.write_text(json.dumps(KEYWORDS, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(path: Path, all_rows, included_rows, excluded_rows):
    lines = []
    lines.append("# AI Hub Whisper Curated Dataset Summary")
    lines.append("")
    lines.append(f"Total label rows: {len(all_rows)}")
    lines.append(f"Included rows: {len(included_rows)}")
    lines.append(f"Excluded rows: {len(excluded_rows)}")
    lines.append("")
    lines.append("## Included By Split/Label")
    for (split, label), count in sorted(Counter((r["split"], r["label"]) for r in included_rows).items()):
        lines.append(f"- {split} / {label}: {count}")
    lines.append("")
    lines.append("## Included By Category")
    for category, count in sorted(Counter(r["category_02"] for r in included_rows).items()):
        lines.append(f"- {category}: {count}")
    lines.append("")
    lines.append("## Top Included Transcripts")
    for transcript, count in Counter(r["transcript"] for r in included_rows).most_common(40):
        lines.append(f"- {transcript}: {count}")
    lines.append("")
    lines.append("## Excluded By Category")
    for category, count in sorted(Counter(r["category_02"] for r in excluded_rows).items()):
        lines.append(f"- {category}: {count}")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_zip_from_tar(tar_path: Path, temp_dir: Path) -> Path:
    out_zip = temp_dir / (tar_path.stem + ".zip")
    with tarfile.open(tar_path, "r:*") as tf, out_zip.open("wb") as out:
        members = [m for m in tf.getmembers() if m.isfile()]
        members.sort(key=lambda m: part_offset(m.name))
        for member in members:
            fh = tf.extractfile(member)
            if fh is None:
                continue
            shutil.copyfileobj(fh, out, length=1024 * 1024 * 8)
    return out_zip


def extract_needed_audio(raw_dir: Path, out_dir: Path, included_rows):
    wanted_by_category = defaultdict(dict)
    for row in included_rows:
        wanted_by_category[(row["split"], row["_source_category"])][row["audio_id"]] = row

    audio_root = out_dir / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="aihub_zip_", dir=out_dir) as tmp:
        temp_dir = Path(tmp)
        for (split, category), wanted in sorted(wanted_by_category.items()):
            if not wanted:
                continue
            tar_name = FILES[(split, category, "audio")]
            tar_path = raw_dir / tar_name
            print(f"[audio] rebuilding zip from {tar_name} for {len(wanted)} requested files", flush=True)
            zip_path = build_zip_from_tar(tar_path, temp_dir)
            extracted = 0
            with zipfile.ZipFile(zip_path) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    decoded = decode_zip_name(info.filename)
                    if not decoded.lower().endswith(".wav"):
                        continue
                    stem = Path(decoded).stem
                    row = wanted.get(stem)
                    if row is None:
                        continue
                    dest = audio_root / row["split"] / row["label"] / f"{stem}.wav"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, dest.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024 * 4)
                    row["wav_path"] = str(dest)
                    extracted += 1
            print(f"[audio] extracted {extracted}/{len(wanted)} from {tar_name}", flush=True)
            zip_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--labels-only", action="store_true")
    args = parser.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in FILES.values() if not (raw_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing required archives: {missing}")

    all_rows = list(iter_label_rows(raw_dir))
    included_rows = [r for r in all_rows if r.get("_include")]
    excluded_rows = [r for r in all_rows if not r.get("_include")]

    write_keywords(out_dir / "help_keywords.json")
    write_csv(out_dir / "manifest_labels_all.csv", all_rows)
    write_csv(out_dir / "manifest_excluded.csv", excluded_rows)

    if not args.labels_only:
        extract_needed_audio(raw_dir, out_dir, included_rows)

    write_csv(out_dir / "manifest.csv", included_rows)
    write_summary(out_dir / "dataset_summary.md", all_rows, included_rows, excluded_rows)

    print(f"[done] all labels: {len(all_rows)}", flush=True)
    print(f"[done] included: {len(included_rows)}", flush=True)
    print(f"[done] excluded: {len(excluded_rows)}", flush=True)
    print(f"[done] output: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
