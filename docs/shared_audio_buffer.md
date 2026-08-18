# Whisper와 AST의 공용 오디오 버퍼

## 목적

Whisper 음성 인식 모델과 AST 환경음 분류 모델을 동시에 사용할 때 마이크를 모델마다 따로 열지 않는다. 마이크 입력은 한 번만 받고, RAM에 있는 공용 오디오 버퍼에서 각 모델이 필요한 시간 구간을 가져간다.

```text
USB 마이크
    |
    v
16 kHz mono 원본 음성
    |
    v
공용 순환 오디오 버퍼
    |-- 최근 5초  --> Whisper 전처리 --> 음성 인식
    `-- 최근 10초 --> AST 전처리     --> 환경음 분류
```

이 구조를 사용하면 두 모델이 같은 시간대의 소리를 분석하므로, 낙상음과 위급 음성이 연속으로 발생했는지 정확하게 연결할 수 있다.

## 공통 입력 규격

현재 프로젝트의 권장 공통 입력은 다음과 같다.

| 항목 | 값 |
|---|---|
| 샘플레이트 | 16,000 Hz |
| 채널 | mono |
| 실시간 메모리 형식 | float32 파형 |
| 파일로 저장할 때 | WAV, 16-bit PCM |
| 공용 버퍼 크기 | 최근 30초 |
| Whisper 입력 구간 | 최근 5초 |
| AST 입력 구간 | 모델 설정에 따라 약 10초 |

AST가 별도의 샘플레이트로 파인튜닝됐다면 해당 모델의 `preprocessor_config.json`을 우선 적용해야 한다.

## 버퍼 동작 방식

마이크는 짧은 음성 조각을 계속 전달한다. 새 음성은 버퍼 뒤에 추가하고, 설정된 최대 길이를 초과하면 가장 오래된 음성을 제거한다. 이런 고정 크기 구조를 순환 버퍼 또는 링 버퍼라고 한다.

현재 시간이 35초이고 버퍼 크기가 30초라면 다음과 같이 동작한다.

```text
공용 버퍼: [5초 -------------------------------- 35초]
Whisper:                                  [30초 - 35초]
AST:                                [25초 ------- 35초]
```

두 모델은 공용 버퍼를 직접 수정하지 않는다. 추론을 시작할 때 필요한 구간의 작은 복사본을 가져가고, 각자 전처리한 뒤 모델에 입력한다.

## 모델별 전처리

두 모델은 동일한 16 kHz mono 파형을 받을 수 있지만 최종 모델 입력은 서로 다르다.

| 구분 | Whisper | AST-base |
|---|---|---|
| 원본 입력 | 16 kHz mono 파형 | 16 kHz mono 파형 |
| 특징 | Log-Mel spectrogram | Log-Mel spectrogram |
| Mel 구간 수 | 80 | 일반적으로 128 |
| 기본 처리 길이 | 최대 30초 | 일반적으로 약 10초 |
| 정규화 | Whisper 전용 | AST/AudioSet 전용 |

따라서 Whisper에서 만든 80-bin Log-Mel 특징을 AST에 넣거나, AST에서 만든 128-bin Log-Mel 특징을 Whisper에 넣으면 안 된다. 공통으로 공유하는 것은 전처리 전의 원본 파형이다.

## 메모리 사용량

16 kHz mono float32 파형은 샘플 하나가 4바이트이므로 다음 정도의 메모리를 사용한다.

```text
1초  = 16,000 x 4 bytes = 약 64 KB
5초  = 약 320 KB
10초 = 약 640 KB
30초 = 약 1.92 MB
```

오디오 버퍼의 메모리 사용량은 모델 가중치보다 매우 작다. 실제 메모리 대부분은 Whisper와 AST 모델 및 추론 중간 계산값이 차지한다.

## 구현 개념

```python
audio_buffer = SharedAudioBuffer(
    sample_rate=16000,
    channels=1,
    max_seconds=30,
)


def microphone_callback(new_audio):
    # 마이크 콜백에서는 버퍼 기록만 수행한다.
    audio_buffer.write(new_audio)


microphone.start(callback=microphone_callback)

# 동일한 시간축을 기준으로 서로 다른 길이를 가져온다.
whisper_audio = audio_buffer.get_latest(seconds=5)
ast_audio = audio_buffer.get_latest(seconds=10)

speech_result = whisper.predict(whisper_audio)
sound_result = ast.predict(ast_audio)
```

실제 구현에서는 마이크가 쓰는 도중 모델이 읽지 않도록 짧은 잠금(lock)을 사용한다. 마이크 콜백 안에서 모델 추론을 실행하면 녹음이 끊길 수 있으므로, 추론은 별도의 작업 스레드에서 실행한다.

## 실시간 처리 권장 구조

1. 마이크 캡처 작업은 하나만 실행한다.
2. 마이크 콜백은 음성을 버퍼에 기록하고 즉시 반환한다.
3. AST 작업은 정해진 간격으로 최근 환경음을 검사한다.
4. Whisper 작업은 음성 구간 또는 위급 상황 후보가 감지됐을 때 실행한다.
5. 모든 결과에 시작 시각과 종료 시각을 기록한다.
6. 처리가 밀리면 오래된 추론 요청을 버리고 최신 오디오를 우선한다.

예상 출력 형식은 다음처럼 통일할 수 있다.

```json
{
  "start_time": 25.0,
  "end_time": 35.0,
  "speech": {
    "text": "살려주세요",
    "label": "help_direct"
  },
  "sound": {
    "label": "fall_sound",
    "score": 0.91
  },
  "emergency_level": "HIGH"
}
```

핵심은 마이크와 원본 음성을 공유하고, 모델별 특징 추출과 추론은 독립적으로 수행하는 것이다.
