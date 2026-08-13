import argparse
import csv
import inspect
import os
from pathlib import Path
from typing import Any


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def row_sort_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("label", ""), row.get("audio_id", ""))


def select_balanced_rows(rows: list[dict[str, str]], limit: int | None) -> list[dict[str, str]]:
    if not limit or limit <= 0 or len(rows) <= limit:
        return rows

    by_label: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_label.setdefault(row.get("label", ""), []).append(row)

    labels = sorted(by_label)
    base_quota = limit // max(1, len(labels))
    remainder = limit % max(1, len(labels))

    selected: list[dict[str, str]] = []
    selected_keys: set[tuple[str, str]] = set()
    for index, label in enumerate(labels):
        quota = base_quota + (1 if index < remainder else 0)
        for row in by_label[label][:quota]:
            selected.append(row)
            selected_keys.add((row.get("label", ""), row.get("audio_id", "")))

    if len(selected) < limit:
        for row in rows:
            key = (row.get("label", ""), row.get("audio_id", ""))
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)
            if len(selected) == limit:
                break

    return sorted(selected, key=row_sort_key)


def load_manifest_rows(manifest: Path, split: str, limit: int | None) -> list[dict[str, str]]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = [row for row in csv.DictReader(fh) if row.get("split") == split]
    rows.sort(key=row_sort_key)
    return select_balanced_rows(rows, limit)


def require_audio_paths(rows: list[dict[str, str]]) -> None:
    missing = [row.get("wav_path", "") for row in rows if not Path(row.get("wav_path", "")).is_file()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} audio files. First missing paths:\n{preview}")


def build_lora_config(r: int, alpha: int, dropout: float, target_modules: list[str]) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
    )


class WhisperManifestDataset:
    def __init__(self, rows: list[dict[str, str]], processor: Any):
        self.rows = rows
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import soundfile as sf

        row = self.rows[index]
        audio, sample_rate = sf.read(row["wav_path"], dtype="float32")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            raise ValueError(f"Expected 16000 Hz audio, got {sample_rate}: {row['wav_path']}")

        features = self.processor.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="np",
        ).input_features[0]
        labels = self.processor.tokenizer(row["transcript"]).input_ids
        return {"input_features": features, "labels": labels}


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        input_features = [{"input_features": item["input_features"]} for item in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": item["labels"]} for item in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        decoder_start_token_id = self.processor.tokenizer.bos_token_id
        if labels.shape[1] > 0 and (labels[:, 0] == decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels.to(torch.long)
        return batch


def build_training_arguments(args: argparse.Namespace, fp16: bool) -> Any:
    from transformers import Seq2SeqTrainingArguments

    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    eval_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps if args.max_steps > 0 else -1,
        "fp16": fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "predict_with_generate": False,
        "generation_max_length": 64,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "dataloader_num_workers": 0,
        "save_total_limit": args.save_total_limit,
        "report_to": [],
        "optim": "adamw_torch",
    }
    kwargs[eval_key] = "steps"
    return Seq2SeqTrainingArguments(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(r"C:\rp5\whisper\models\whisper-small-ko-zeroth"))
    parser.add_argument("--manifest", type=Path, default=Path("whisper_curated_processed/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("whisper_models/whisper-small-ko-fall-help-lora"))
    parser.add_argument("--merge-output-dir", type=Path, default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--language", default="Korean")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--skip-audio-check", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from peft import get_peft_model
    from transformers import Seq2SeqTrainer, WhisperForConditionalGeneration, WhisperProcessor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_manifest_rows(args.manifest, args.train_split, args.train_limit or None)
    eval_rows = load_manifest_rows(args.manifest, args.eval_split, args.eval_limit or None)
    if not train_rows:
        raise SystemExit(f"No rows found for split={args.train_split}")
    if not eval_rows:
        raise SystemExit(f"No rows found for split={args.eval_split}")
    if not args.skip_audio_check:
        require_audio_paths(train_rows)
        require_audio_paths(eval_rows)

    cuda_available = torch.cuda.is_available()
    fp16 = cuda_available and not args.no_fp16
    dtype = torch.float16 if fp16 else torch.float32
    processor = WhisperProcessor.from_pretrained(
        str(args.model),
        language=args.language,
        task=args.task,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        str(args.model),
        torch_dtype=dtype,
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    gradient_checkpointing = not args.no_gradient_checkpointing
    args.gradient_checkpointing = gradient_checkpointing
    if gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = build_lora_config(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = WhisperManifestDataset(train_rows, processor)
    eval_dataset = WhisperManifestDataset(eval_rows, processor)
    trainer = Seq2SeqTrainer(
        args=build_training_arguments(args, fp16=fp16),
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        processing_class=processor,
    )

    print(
        f"[train] train_rows={len(train_rows)} eval_rows={len(eval_rows)} "
        f"cuda={cuda_available} fp16={fp16} output={args.output_dir}",
        flush=True,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    if args.merge_output_dir:
        args.merge_output_dir.mkdir(parents=True, exist_ok=True)
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(args.merge_output_dir), safe_serialization=True)
        processor.save_pretrained(str(args.merge_output_dir))
        print(f"[merge] saved merged model to {args.merge_output_dir}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
