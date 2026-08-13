import argparse
import csv
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_MODULE_PATH = REPO_ROOT / "tools" / "evaluate_whisper_baseline.py"


def load_eval_module():
    spec = importlib.util.spec_from_file_location("evaluate_whisper_baseline", EVAL_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--shard-dir", required=True, action="append", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for shard_dir in args.shard_dir:
        predictions = shard_dir / "baseline_predictions.csv"
        if not predictions.exists():
            raise SystemExit(f"missing shard predictions: {predictions}")
        rows.extend(read_predictions(predictions))

    rows.sort(key=lambda row: (row.get("label", ""), row.get("audio_id", ""), row.get("wav_path", "")))
    module = load_eval_module()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    module.write_csv(args.out_dir / "baseline_predictions.csv", rows)
    summary = module.summarize(rows)
    (args.out_dir / "baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
