import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def split_rows(
    rows: list[dict[str, str]],
    train_ratio: float,
    seed: int,
    source_split: str | None = None,
    holdout_split: str | None = None,
    holdout_name: str = "test",
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if source_split and row.get("split") != source_split:
            continue
        grouped[row.get("label", "")].append(row)

    output: list[dict[str, str]] = []
    for label in sorted(grouped):
        label_rows = grouped[label]
        rng.shuffle(label_rows)
        train_count = round(len(label_rows) * train_ratio)
        for index, row in enumerate(label_rows):
            copied = dict(row)
            copied["split"] = "train" if index < train_count else "validation"
            output.append(copied)

    if holdout_split:
        for row in rows:
            if row.get("split") != holdout_split:
                continue
            copied = dict(row)
            copied["split"] = holdout_name
            output.append(copied)

    output.sort(key=lambda row: (row.get("split", ""), row.get("label", ""), row.get("audio_id", "")))
    return output


def summarize(rows: list[dict[str, str]]) -> dict:
    by_split_label = Counter((row.get("split", ""), row.get("label", "")) for row in rows)
    return {
        "total": len(rows),
        "by_split": dict(Counter(row.get("split", "") for row in rows)),
        "by_label": dict(Counter(row.get("label", "") for row in rows)),
        "by_split_label": {f"{split}/{label}": count for (split, label), count in sorted(by_split_label.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("whisper_curated_processed/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("whisper_curated_processed/manifest_8_2.csv"))
    parser.add_argument("--summary", type=Path, default=Path("whisper_curated_processed/manifest_8_2_summary.json"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-split", default=None)
    parser.add_argument("--holdout-split", default=None)
    parser.add_argument("--holdout-name", default="test")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    output_rows = split_rows(
        rows,
        args.train_ratio,
        args.seed,
        source_split=args.source_split,
        holdout_split=args.holdout_split,
        holdout_name=args.holdout_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = summarize(output_rows)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
