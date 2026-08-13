import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


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


def merged_fieldnames(*field_groups: list[str]) -> list[str]:
    fields: list[str] = []
    for group in field_groups:
        for field in group:
            if field not in fields:
                fields.append(field)
    return fields


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


def row_key(row: dict[str, str]) -> str:
    return row.get("audio_id") or row.get("wav_path") or json.dumps(row, ensure_ascii=False, sort_keys=True)


def repeated_user_rows(rows: list[dict[str, str]], repeat: int) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for repeat_index in range(1, repeat + 1):
        for row in rows:
            copied = dict(row)
            copied["split"] = "train"
            copied["audio_id"] = f"{row.get('audio_id', 'user_recording')}_rep{repeat_index:02d}"
            copied["category_01"] = row.get("category_01") or "user_recording"
            copied["category_03"] = "user_recording_oversampled"
            output.append(copied)
    return output


def summarize(rows: list[dict[str, str]]) -> dict:
    by_split_label = Counter((row.get("split", ""), row.get("label", "")) for row in rows)
    by_source = Counter(row.get("category_01", "") for row in rows)
    return {
        "total": len(rows),
        "by_split": dict(Counter(row.get("split", "") for row in rows)),
        "by_label": dict(Counter(row.get("label", "") for row in rows)),
        "by_split_label": {f"{split}/{label}": count for (split, label), count in sorted(by_split_label.items())},
        "by_source": dict(by_source),
        "top_transcripts": dict(Counter(row.get("transcript", "") for row in rows).most_common(30)),
    }


def write_summary(path: Path, summary: dict, args: argparse.Namespace) -> None:
    lines = [
        "# Fine-tune Mix Manifest Summary",
        "",
        f"AI Hub train sample limit: {args.aihub_train_limit}",
        f"AI Hub validation sample limit: {args.aihub_validation_limit}",
        f"User recording repeat: {args.user_repeat}",
        f"Seed: {args.seed}",
        "",
        f"Total rows: {summary['total']}",
        "",
        "## By Split",
    ]
    for split, count in sorted(summary["by_split"].items()):
        lines.append(f"- {split}: {count}")
    lines.extend(["", "## By Split/Label"])
    for key, count in summary["by_split_label"].items():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## By Source"])
    for source, count in sorted(summary["by_source"].items()):
        lines.append(f"- {source or '(blank)'}: {count}")
    lines.extend(["", "## Top Transcripts"])
    for transcript, count in summary["top_transcripts"].items():
        lines.append(f"- {transcript}: {count}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aihub-manifest", type=Path, default=Path("whisper_curated_processed/manifest_train8_val2_test_with_112.csv"))
    parser.add_argument("--user-manifest", type=Path, default=Path("whisper_user_recordings_processed/manifest_all_test.csv"))
    parser.add_argument("--output", type=Path, default=Path("whisper_user_recordings_processed/manifest_aihub5000_userx5.csv"))
    parser.add_argument("--summary", type=Path, default=Path("whisper_user_recordings_processed/manifest_aihub5000_userx5_summary.md"))
    parser.add_argument("--aihub-train-limit", type=int, default=5000)
    parser.add_argument("--aihub-validation-limit", type=int, default=1000)
    parser.add_argument("--user-repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-aihub-test", action="store_true")
    args = parser.parse_args()

    aihub_fields, aihub_rows = read_csv(args.aihub_manifest)
    user_fields, user_rows = read_csv(args.user_manifest)

    aihub_train = sample_balanced(
        [row for row in aihub_rows if row.get("split") == "train"],
        args.aihub_train_limit,
        args.seed,
    )
    aihub_validation = sample_balanced(
        [row for row in aihub_rows if row.get("split") == "validation"],
        args.aihub_validation_limit,
        args.seed + 1,
    )
    for row in aihub_train:
        row["split"] = "train"
    for row in aihub_validation:
        row["split"] = "validation"

    mixed_rows = aihub_train + repeated_user_rows(user_rows, args.user_repeat) + aihub_validation
    if args.include_aihub_test:
        mixed_rows.extend(row for row in aihub_rows if row.get("split") == "test")

    mixed_rows.sort(key=lambda row: (row.get("split", ""), row.get("label", ""), row.get("transcript", ""), row.get("audio_id", "")))
    fields = merged_fieldnames(aihub_fields, user_fields)
    write_csv(args.output, fields, mixed_rows)
    summary = summarize(mixed_rows)
    write_summary(args.summary, summary, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
