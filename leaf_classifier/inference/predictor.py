import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms


def _build_eval_transform(cfg: DictConfig) -> transforms.Compose:
    image_size = tuple(cfg.data.image_size)
    norm = cfg.data.normalize
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(norm.mean), std=list(norm.std)),
        ]
    )


def _load_class_names(artifacts_dir: Path) -> list[str]:
    with open(artifacts_dir / "class_names.json") as f:
        return json.load(f)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()


def _decode_top_k(probabilities: np.ndarray, class_names: list[str], top_k: int) -> list[dict]:
    indices = np.argsort(probabilities)[::-1][:top_k]
    return [
        {"class": class_names[idx], "probability": float(probabilities[idx])} for idx in indices
    ]


def _pull_dvc_artifact(path: Path) -> None:
    try:
        from dvc.repo import Repo

        with Repo() as repo:
            repo.pull(targets=[str(path)])
    except Exception:
        pass


def run_inference(cfg: DictConfig, image_path: Path) -> dict:
    artifacts_dir = Path(cfg.data.artifacts_dir)
    onnx_path = artifacts_dir / f"model_{cfg.model.name}.onnx"

    _pull_dvc_artifact(onnx_path)

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {onnx_path}. "
            f"Run 'python commands.py train model={cfg.model.name}' first."
        )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    class_names = _load_class_names(artifacts_dir)
    transform = _build_eval_transform(cfg)

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).numpy()

    input_name = session.get_inputs()[0].name
    raw_output = session.run(None, {input_name: image_tensor})[0][0]

    probabilities = _softmax(raw_output)
    top3 = _decode_top_k(probabilities, class_names, top_k=3)
    return {"top1": top3[0], "top3": top3}
