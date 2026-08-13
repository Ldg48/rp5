import argparse
import os
import sys
import tempfile


def check_ffmpeg():
    ffmpeg_paths = [
        "c:/ffmpeg/bin/ffmpeg.exe",
        "C:/ffmpeg/bin/ffmpeg.exe",
        "ffmpeg.exe",
    ]
    for path in ffmpeg_paths:
        if os.path.exists(path):
            os.environ["PATH"] = os.path.dirname(path) + ";" + os.environ.get("PATH", "")
            return True
    sys.stderr.write(
        "ERROR: FFmpeg를 찾을 수 없습니다.\n"
        "다음 경로에 설치되어 있는지 확인하세요:\n"
        "- c:/ffmpeg/bin/ffmpeg.exe\n"
        "- 시스템 PATH\n"
    )
    return False


def record_microphone_audio(output_path, duration, samplerate):
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write
    except ModuleNotFoundError as exc:
        raise RuntimeError("sounddevice와 scipy가 필요합니다. pip install sounddevice scipy") from exc

    print(f"녹음 시작: {duration:.1f}초, 샘플레이트 {samplerate}Hz")
    try:
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
        sd.wait()
        write(output_path, samplerate, recording)
    except Exception as exc:
        raise RuntimeError(f"마이크 녹음 중 오류가 발생했습니다: {exc}") from exc


def transcribe_whisper(audio_path, model_name, language):
    try:
        import whisper
    except ModuleNotFoundError as exc:
        raise RuntimeError("openai-whisper 패키지가 필요합니다. pip install openai-whisper") from exc

    print(f"Whisper 모델 로드 중: {model_name}")
    model = whisper.load_model(model_name)
    print("Whisper 모델 로드 완료.")

    return model.transcribe(audio_path, language=language, fp16=False).get("text", "")


def parse_args():
    parser = argparse.ArgumentParser(description="Whisper small/medium 마이크 또는 오디오 파일 테스트")
    parser.add_argument("--audio", help="테스트할 오디오 파일 경로")
    parser.add_argument("--mic", action="store_true", help="마이크로 녹음하여 테스트")
    parser.add_argument("--duration", type=float, default=5.0, help="마이크 녹음 길이(초)")
    parser.add_argument("--samplerate", type=int, default=16000, help="녹음 샘플레이트")
    parser.add_argument(
        "--method",
        default="all",
        choices=["all", "whisper-small", "whisper-medium"],
        help="실행할 ASR 방법",
    )
    parser.add_argument("--whisper-model", default="medium", help="Whisper 모델 이름")
    parser.add_argument("--language", default="ko", help="Whisper에 사용할 언어 코드")
    parser.add_argument("--output", default=None, help="출력 텍스트 파일")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.audio and not args.mic:
        sys.stderr.write("--audio 또는 --mic 중 하나를 지정해야 합니다.\n")
        return 1

    if not check_ffmpeg():
        return 1

    audio_path = args.audio
    temp_file = None
    if args.mic:
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()
        audio_path = temp_file.name
        try:
            record_microphone_audio(audio_path, args.duration, args.samplerate)
            print(f"녹음 완료: {audio_path}")
        except RuntimeError as exc:
            sys.stderr.write(str(exc) + "\n")
            return 1

    results = {}
    try:
        methods = [args.method] if args.method != "all" else [
            "whisper-small",
            "whisper-medium",
        ]

        for method in methods:
            print("\n" + "=" * 50)
            print(f"실행 중: {method}")
            try:
                model_name = "small" if method == "whisper-small" else args.whisper_model
                text = transcribe_whisper(audio_path, model_name, args.language)
                results[method] = text.strip()
                print("결과:\n" + results[method])
            except Exception as exc:
                results[method] = f"ERROR: {exc}"
                print(f"{method} 오류: {exc}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                for method, text in results.items():
                    f.write(f"=== {method} ===\n")
                    f.write(text + "\n\n")
            print(f"모든 출력 결과를 저장했습니다: {args.output}")

        return 0
    finally:
        if temp_file is not None:
            try:
                os.remove(temp_file.name)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
