import argparse
import shutil
import sys
import tempfile
import os

import sounddevice as sd
from scipy.io.wavfile import write


def check_ffmpeg():
    # FFmpeg 경로 직접 지정
    ffmpeg_paths = [
        "c:/ffmpeg/bin/ffmpeg.exe",
        "C:/ffmpeg/bin/ffmpeg.exe",
        "ffmpeg.exe"
    ]
    for path in ffmpeg_paths:
        if os.path.exists(path):
            os.environ['PATH'] = os.path.dirname(path) + ';' + os.environ.get('PATH', '')
            return True
    sys.stderr.write(
        "ERROR: FFmpeg를 찾을 수 없습니다.\n"
        "다음 경로에 설치되어 있는지 확인하세요:\n"
        "- c:/ffmpeg/bin/ffmpeg.exe\n"
        "- 시스템 PATH\n"
    )
    return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Whisper로 노트북 마이크 입력을 녹음하고 텍스트로 변환합니다."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="녹음 길이(초), 기본값 5초"
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper 모델 이름 (tiny, base, small, medium, large)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="변환 결과를 저장할 텍스트 파일 경로"
    )
    parser.add_argument(
        "--language",
        default=None,
        help="오디오 언어를 지정하려면 코드 입력 (예: ko, en)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="사용할 마이크 장치 이름 또는 ID (선택)"
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=16000,
        help="녹음 샘플레이트, 기본 16000"
    )
    return parser.parse_args()


def list_devices():
    print("=== 사용 가능한 오디오 장치 목록 ===")
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        default = "(default)" if idx == sd.default.device[0] or idx == sd.default.device[1] else ""
        print(f"{idx}: {dev['name']}  {dev['max_input_channels']} input {default}")
    print("===================================\n")


def main():
    args = parse_args()

    if not check_ffmpeg():
        return 1

    try:
        import whisper
    except ModuleNotFoundError:
        sys.stderr.write(
            "ERROR: openai-whisper 패키지가 설치되어 있지 않습니다.\n"
            "pip install openai-whisper 를 실행하세요.\n"
        )
        return 1

    if args.device is not None:
        try:
            sd.default.device = int(args.device)
        except ValueError:
            sd.default.device = args.device

    try:
        sd.check_input_settings(device=sd.default.device, samplerate=args.samplerate, channels=1)
    except Exception as exc:
        sys.stderr.write("마이크 입력 설정 오류:\n")
        sys.stderr.write(str(exc) + "\n")
        list_devices()
        return 1

    print(f"녹음 시작: {args.duration:.1f}초, 샘플레이트 {args.samplerate}Hz")
    print("마이크에서 말하세요...")
    try:
        recording = sd.rec(int(args.duration * args.samplerate), samplerate=args.samplerate, channels=1, dtype='int16')
        sd.wait()
    except Exception as exc:
        sys.stderr.write("녹음 중 오류가 발생했습니다:\n")
        sys.stderr.write(str(exc) + "\n")
        return 1

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        wav_path = tmp.name
    write(wav_path, args.samplerate, recording)
    print(f"녹음 완료: {wav_path}")

    print(f"Whisper 모델 로드 중: {args.model}")
    model = whisper.load_model(args.model)
    print("모델 로드 완료.")

    print("변환 중...")
    result = model.transcribe(wav_path, language=args.language, fp16=False)
    print("\n=== Whisper 결과 ===")
    print("감지된 언어:", result.get("language"))
    print("텍스트:")
    print(result.get("text", ""))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result.get("text", ""))
        print(f"결과를 저장했습니다: {args.output}")

    os.remove(wav_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())