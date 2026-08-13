# Whisper / Korean ASR Tools

이 폴더에는 Whisper 및 기타 ASR 프레임워크를 테스트하는 데 사용할 수 있는 스크립트가 포함되어 있습니다.

## 설치

```powershell
cd c:\rp5\whisper
pip install -r requirements.txt
```

## 스크립트

- `demo.py`
  - 오디오 파일을 Whisper로 변환합니다.
  - 예: `python demo.py --audio ../samples/ko.wav --model medium --language ko`

- `mic.py`
  - 마이크 입력을 녹음하고 Whisper로 변환합니다.
  - 예: `python mic.py --duration 8 --model small --language ko`

- `model_test.py`
  - Whisper 모델 `tiny`, `base`, `small`, `medium`의 로드 및 처리 속도를 비교합니다.
  - 예: `python model_test.py`

- `multi_asr.py`
  - Whisper `small`과 `medium` 모델을 파일 또는 마이크 입력으로 테스트합니다.
  - 예: `python multi_asr.py --audio ../samples/ko.wav --method all --output multi_results.txt`

## 권장 모델

- Whisper `small`
- Whisper `medium`
