import argparse
import os
import shutil
import sys


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
        description="Whisper 음성 인식 데모 스크립트"
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="변환할 오디오 파일 경로 (wav/mp3/m4a/flac 등)"
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper 모델 이름 (tiny, base, small, medium, large)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="결과 텍스트를 저장할 파일 경로 (선택)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="입력 오디오 언어를 지정하려면 코드(e.g. ko, en) 사용 (선택)",
    )
    return parser.parse_args()


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

    print(f"모델 로드 중: {args.model}")
    model = whisper.load_model(args.model)
    print("모델 로드 완료.")

    print(f"오디오 파일 변환 시작: {args.audio}")
    result = model.transcribe(args.audio, language=args.language, fp16=False)

    print("\n=== Whisper 결과 ===")
    print("감지된 언어:", result.get("language"))
    print("텍스트:")
    print(result.get("text", ""))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result.get("text", ""))
        print(f"결과를 저장했습니다: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())