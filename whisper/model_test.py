import time
import whisper
import numpy as np

# 테스트용 샘플 오디오 생성 (랜덤 노이즈로 한국어 음성 시뮬레이션)
sample_rate = 16000
duration = 3  # 3초

np.random.seed(42)  # 재현성을 위해
audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

models = ['tiny', 'base', 'small', 'medium']
results = {}

for model_name in models:
    print(f'\n=== {model_name.upper()} 모델 테스트 ===')

    # 모델 로드 시간 측정
    print(f'{model_name} 모델 로드 중...')
    load_start = time.time()
    model = whisper.load_model(model_name)
    load_time = time.time() - load_start
    print(f'로드 시간: {load_time:.1f}초')

    # 오디오 처리 시간 측정
    print(f'{duration}초 오디오 변환 중...')
    process_start = time.time()
    result = model.transcribe(audio, language='ko', fp16=False)
    process_time = time.time() - process_start

    results[model_name] = {
        'load_time': load_time,
        'process_time': process_time,
        'detected_lang': result.get('language'),
        'text': result.get('text', '').strip()
    }

    print(f'처리 시간: {process_time:.1f}초')
    print(f'감지 언어: {result.get("language")}')
    print(f'텍스트: "{result.get("text", "").strip()}"')

print('\n' + '='*50)
print('최종 비교 결과')
print('='*50)
for model, data in results.items():
    print(f'{model.upper()}:')
    print(f'  로드 시간: {data["load_time"]:.1f}초')
    print(f'  처리 시간: {data["process_time"]:.1f}초')
    print(f'  감지 언어: {data["detected_lang"]}')
    print(f'  텍스트 길이: {len(data["text"])} 글자')
    print()