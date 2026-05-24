import json
from pathlib import Path

import fire
import torch
from hydra import compose, initialize_config_dir

from leaf_classifier.models.baseline import BaselineCNN
from leaf_classifier.models.cnn import PlantDiseaseCNN
from leaf_classifier.utils.export import export_to_onnx


def convert(checkpoint_path: str, **kwargs) -> None:
    config_dir = str(Path(__file__).parent.parent / "configs")
    overrides = [f"{key}={value}" for key, value in kwargs.items()]
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=overrides)

    artifacts_dir = Path(cfg.data.artifacts_dir)
    class_names_path = artifacts_dir / "class_names.json"
    if not class_names_path.exists():
        raise FileNotFoundError(
            f"{class_names_path} not found. Run training first to generate artifacts."
        )
    with open(class_names_path) as f:
        num_classes = len(json.load(f))

    ckpt_data = torch.load(checkpoint_path, map_location="cpu")
    inner_state = {
        key[len("model.") :]: value
        for key, value in ckpt_data["state_dict"].items()
        if key.startswith("model.")
    }

    if cfg.model.name == "baseline":
        model = BaselineCNN(num_classes=num_classes, dropout_rate=cfg.model.dropout_rate)
    else:
        model = PlantDiseaseCNN(num_classes=num_classes, dropout_rate=cfg.model.dropout_rate)

    model.load_state_dict(inner_state)

    onnx_path = export_to_onnx(model, cfg)
    print(f"[INFO] Exported ONNX model to: {onnx_path}")


if __name__ == "__main__":
    fire.Fire(convert)
