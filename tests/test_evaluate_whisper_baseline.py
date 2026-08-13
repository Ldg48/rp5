import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "evaluate_whisper_baseline.py"


def load_evaluate_module():
    spec = importlib.util.spec_from_file_location("evaluate_whisper_baseline", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transformers_generate_kwargs_limit_repetition_for_short_emergency_phrases():
    module = load_evaluate_module()

    kwargs = module.build_generate_kwargs(language="ko", max_new_tokens=10)

    assert kwargs == {
        "language": "ko",
        "task": "transcribe",
        "max_new_tokens": 10,
        "num_beams": 1,
        "do_sample": False,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
    }
