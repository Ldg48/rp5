import csv
import json
from pathlib import Path


ROOT = Path("compare_eval_7_3_20260729")
REPORT_MD = ROOT / "evaluation_report.md"
REPORT_CSV = ROOT / "evaluation_table.csv"

MODELS = [
    {
        "name": "Zeroth baseline HF",
        "run": "GPU / HF Transformers / max_new_tokens=10",
        "result_dir": ROOT / "zeroth_hf",
        "model_path": Path("C:/rp5/whisper/models/whisper-small-ko-zeroth"),
    },
    {
        "name": "Zeroth baseline INT8",
        "run": "CPU / CTranslate2 INT8 / beam_size=1",
        "result_dir": ROOT / "zeroth_int8",
        "model_path": Path("whisper_models/whisper-small-ko-zeroth-ct2-int8"),
    },
    {
        "name": "LoRA fine-tuned HF",
        "run": "GPU / HF Transformers / max_new_tokens=10",
        "result_dir": ROOT / "lora_userx5_hf",
        "model_path": Path("whisper_models/whisper-small-ko-fall-help-userx5-lora-merged"),
    },
    {
        "name": "LoRA fine-tuned INT8",
        "run": "CPU / CTranslate2 INT8 / beam_size=1",
        "result_dir": ROOT / "lora_userx5_int8",
        "model_path": Path("whisper_models/whisper-small-ko-fall-help-userx5-ct2-int8"),
    },
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def model_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / 1024 / 1024
    total = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
    return total / 1024 / 1024


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {"count": 0, "cer": 0.0, "wer": 0.0, "keyword_hit_rate": 0.0}
    return {
        "count": len(rows),
        "cer": sum(float(row["cer"]) for row in rows) / len(rows),
        "wer": sum(float(row["wer"]) for row in rows) / len(rows),
        "keyword_hit_rate": sum(1 for row in rows if row["keyword_hit"] == "true") / len(rows),
    }


def source_name(row: dict[str, str]) -> str:
    return "New direct recording" if row.get("category_01") == "new_eval_recording" else "AI Hub held-out"


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_float(value: float) -> str:
    return f"{value:.4f}"


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def main() -> int:
    csv_rows: list[dict[str, str]] = []
    overall_rows: list[list[str]] = []
    source_rows: list[list[str]] = []

    for model in MODELS:
        summary = read_json(model["result_dir"] / "baseline_summary.json")
        predictions = read_predictions(model["result_dir"] / "baseline_predictions.csv")
        size_mb = model_size_mb(model["model_path"])

        overall_rows.append(
            [
                model["name"],
                model["run"],
                str(summary["count"]),
                format_float(summary["cer"]),
                format_float(summary["wer"]),
                format_pct(summary["keyword_hit_rate"]),
                f"{size_mb:.1f} MB",
            ]
        )
        csv_rows.append(
            {
                "model": model["name"],
                "run": model["run"],
                "source": "overall",
                "count": str(summary["count"]),
                "cer": format_float(summary["cer"]),
                "wer": format_float(summary["wer"]),
                "keyword_hit_rate": format_pct(summary["keyword_hit_rate"]),
                "model_size": f"{size_mb:.1f} MB",
            }
        )

        for source in ["AI Hub held-out", "New direct recording"]:
            subset = [row for row in predictions if source_name(row) == source]
            source_summary = summarize_rows(subset)
            source_rows.append(
                [
                    model["name"],
                    source,
                    str(source_summary["count"]),
                    format_float(source_summary["cer"]),
                    format_float(source_summary["wer"]),
                    format_pct(source_summary["keyword_hit_rate"]),
                ]
            )
            csv_rows.append(
                {
                    "model": model["name"],
                    "run": model["run"],
                    "source": source,
                    "count": str(source_summary["count"]),
                    "cer": format_float(source_summary["cer"]),
                    "wer": format_float(source_summary["wer"]),
                    "keyword_hit_rate": format_pct(source_summary["keyword_hit_rate"]),
                    "model_size": f"{size_mb:.1f} MB",
                }
            )

    lines = [
        "# 7:3 Evaluation Report",
        "",
        "Evaluation manifest: `whisper_eval_7_3_20260729/manifest_eval_mixed.csv`",
        "",
        "- Total evaluation rows: 2398",
        "- AI Hub held-out rows: 2370",
        "- New direct recording rows: 28",
        "- Labels: fall_related 1199, help_direct 1199",
        "",
        "## Overall",
    ]
    lines.extend(table(["Model", "Run", "Rows", "CER down", "WER down", "Keyword up", "Size"], overall_rows))
    lines.extend(["", "## By Source"])
    lines.extend(table(["Model", "Source", "Rows", "CER down", "WER down", "Keyword up"], source_rows))
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        fieldnames = ["model", "run", "source", "count", "cer", "wer", "keyword_hit_rate", "model_size"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(REPORT_MD)
    print(REPORT_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
