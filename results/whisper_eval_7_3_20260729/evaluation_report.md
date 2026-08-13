# 7:3 Evaluation Report

Evaluation manifest: `whisper_eval_7_3_20260729/manifest_eval_mixed.csv`

- Total evaluation rows: 2398
- AI Hub held-out rows: 2370
- New direct recording rows: 28
- Labels: fall_related 1199, help_direct 1199

## Overall
| Model | Run | Rows | CER down | WER down | Keyword up | Size |
|---|---|---|---|---|---|---|
| Zeroth baseline HF | GPU / HF Transformers / max_new_tokens=10 | 2398 | 3.3802 | 1.9013 | 91.83% | 924.1 MB |
| Zeroth baseline INT8 | CPU / CTranslate2 INT8 / beam_size=1 | 2398 | 0.4376 | 0.8383 | 88.49% | 239.5 MB |
| LoRA fine-tuned HF | GPU / HF Transformers / max_new_tokens=10 | 2398 | 0.0427 | 0.0482 | 99.62% | 464.6 MB |
| LoRA fine-tuned INT8 | CPU / CTranslate2 INT8 / beam_size=1 | 2398 | 0.0231 | 0.0651 | 99.58% | 234.4 MB |

## By Source
| Model | Source | Rows | CER down | WER down | Keyword up |
|---|---|---|---|---|---|
| Zeroth baseline HF | AI Hub held-out | 2370 | 3.4163 | 1.9158 | 92.24% |
| Zeroth baseline HF | New direct recording | 28 | 0.3221 | 0.6786 | 57.14% |
| Zeroth baseline INT8 | AI Hub held-out | 2370 | 0.4389 | 0.8417 | 88.86% |
| Zeroth baseline INT8 | New direct recording | 28 | 0.3325 | 0.5536 | 57.14% |
| LoRA fine-tuned HF | AI Hub held-out | 2370 | 0.0301 | 0.0468 | 99.87% |
| LoRA fine-tuned HF | New direct recording | 28 | 1.1143 | 0.1607 | 78.57% |
| LoRA fine-tuned INT8 | AI Hub held-out | 2370 | 0.0210 | 0.0616 | 99.83% |
| LoRA fine-tuned INT8 | New direct recording | 28 | 0.1956 | 0.3571 | 78.57% |
