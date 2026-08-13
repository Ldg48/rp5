import csv
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "train_whisper_lora.py"


def load_training_module():
    spec = importlib.util.spec_from_file_location("train_whisper_lora", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_manifest(path: Path, rows: list[dict]) -> None:
    fieldnames = ["split", "label", "audio_id", "transcript", "wav_path"]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_manifest_rows_filters_split_and_sorts(tmp_path):
    module = load_training_module()
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            {"split": "validation", "label": "help_direct", "audio_id": "v1", "transcript": "도와주세요", "wav_path": "v1.wav"},
            {"split": "train", "label": "help_direct", "audio_id": "h2", "transcript": "살려주세요", "wav_path": "h2.wav"},
            {"split": "train", "label": "fall_related", "audio_id": "f1", "transcript": "넘어졌어요", "wav_path": "f1.wav"},
            {"split": "train", "label": "help_direct", "audio_id": "h1", "transcript": "도와주세요", "wav_path": "h1.wav"},
        ],
    )

    rows = module.load_manifest_rows(manifest, split="train", limit=None)

    assert [(row["label"], row["audio_id"]) for row in rows] == [
        ("fall_related", "f1"),
        ("help_direct", "h1"),
        ("help_direct", "h2"),
    ]


def test_load_manifest_rows_uses_balanced_limit(tmp_path):
    module = load_training_module()
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            {"split": "train", "label": "help_direct", "audio_id": "h1", "transcript": "도와주세요", "wav_path": "h1.wav"},
            {"split": "train", "label": "help_direct", "audio_id": "h2", "transcript": "살려주세요", "wav_path": "h2.wav"},
            {"split": "train", "label": "help_direct", "audio_id": "h3", "transcript": "119 불러줘", "wav_path": "h3.wav"},
            {"split": "train", "label": "fall_related", "audio_id": "f1", "transcript": "넘어졌어요", "wav_path": "f1.wav"},
            {"split": "train", "label": "fall_related", "audio_id": "f2", "transcript": "못 일어나겠어요", "wav_path": "f2.wav"},
            {"split": "train", "label": "fall_related", "audio_id": "f3", "transcript": "다쳤어요", "wav_path": "f3.wav"},
        ],
    )

    rows = module.load_manifest_rows(manifest, split="train", limit=4)

    assert len(rows) == 4
    assert [row["label"] for row in rows].count("fall_related") == 2
    assert [row["label"] for row in rows].count("help_direct") == 2


def test_build_lora_config_uses_generic_peft_wrapper_for_whisper():
    module = load_training_module()

    config = module.build_lora_config(
        r=4,
        alpha=8,
        dropout=0.1,
        target_modules=["q_proj"],
    )

    assert config.task_type is None
