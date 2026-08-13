import argparse
import os
import tempfile
from pathlib import Path

import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "whisper_models" / "whisper-small-ko-fall-help-userx5-ct2-int8"


def list_devices() -> None:
    print(sd.query_devices())


def record_wav(path: Path, seconds: float, sample_rate: int, device: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[record] {seconds:.1f}s recording start")
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    print(f"[record] saved: {path}")


def transcribe(model_dir: Path, wav_path: Path, language: str) -> str:
    model = WhisperModel(str(model_dir), device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        str(wav_path),
        language=language,
        task="transcribe",
        beam_size=1,
        without_timestamps=True,
    )
    return "".join(segment.text for segment in segments).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--language", default="ko")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--keep-wav", type=Path, default=None)
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return 0

    if not args.model.exists():
        raise SystemExit(f"[error] model path not found: {args.model.resolve()}")

    wav_path = args.keep_wav
    if wav_path is None:
        temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp.close()
        wav_path = Path(temp.name)

    record_wav(wav_path, args.seconds, args.sample_rate, args.device)
    text = transcribe(args.model, wav_path, args.language)
    print(f"[text] {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
