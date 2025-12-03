# src/utils/model_utils/model_io.py

from pathlib import Path
import joblib


def save_model(model, save_path):
    """
    모델 객체를 joblib으로 저장하는 함수.

    Parameters
    ----------
    model : 학습된 모델 객체
    save_path : str 또는 Path
        저장할 파일 경로 (.pkl 권장)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    print(f"[SAVE] 모델 저장 완료 → {save_path}")


def load_model(load_path):
    """
    저장된 모델을 로드하는 함수.

    Parameters
    ----------
    load_path : str 또는 Path
        로드할 모델 파일 경로

    Returns
    -------
    model : 로드된 모델 객체
    """
    load_path = Path(load_path)
    model = joblib.load(load_path)
    print(f"[LOAD] 모델 로드 완료 ← {load_path}")
    return model
