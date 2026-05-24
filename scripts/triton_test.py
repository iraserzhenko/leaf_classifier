import json
from pathlib import Path

import fire
import numpy as np
from PIL import Image
from torchvision import transforms

DEFAULT_SERVER_URL = "localhost:8000"
DEFAULT_MODEL_NAME = "leaf_classifier_baseline"
IMAGE_SIZE = (256, 256)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def _build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def _load_class_names(artifacts_dir: Path) -> list[str]:
    path = artifacts_dir / "class_names.json"
    if not path.exists():
        raise FileNotFoundError(f"class_names.json not found at {path}")
    with open(path) as f:
        return json.load(f)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()


def test(
    image_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    server_url: str = DEFAULT_SERVER_URL,
    artifacts_dir: str = "artifacts",
    top_k: int = 3,
) -> None:
    try:
        import tritonclient.http as httpclient
    except ImportError as exc:
        raise RuntimeError(
            "tritonclient not installed. Run: uv run pip install 'tritonclient[http]'"
        ) from exc

    class_names = _load_class_names(Path(artifacts_dir))
    transform = _build_transform()

    image = Image.open(image_path).convert("RGB")
    image_array = transform(image).unsqueeze(0).numpy()

    client = httpclient.InferenceServerClient(url=server_url)
    if not client.is_server_live():
        raise RuntimeError(f"Triton server at {server_url} is not live.")
    if not client.is_model_ready(model_name):
        raise RuntimeError(
            f"Model '{model_name}' is not ready on the server. "
            "Available models: leaf_classifier_baseline, leaf_classifier_cnn"
        )

    triton_input = httpclient.InferInput("input", image_array.shape, "FP32")
    triton_input.set_data_from_numpy(image_array)
    triton_output = httpclient.InferRequestedOutput("output")

    response = client.infer(
        model_name=model_name,
        inputs=[triton_input],
        outputs=[triton_output],
    )
    logits = response.as_numpy("output")[0]
    probabilities = _softmax(logits)

    top_indices = np.argsort(probabilities)[::-1][:top_k]
    top_k_results = [
        {"class": class_names[idx], "probability": float(probabilities[idx])} for idx in top_indices
    ]
    result = {"top1": top_k_results[0], "top3": top_k_results}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    fire.Fire(test)
