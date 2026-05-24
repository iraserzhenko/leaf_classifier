from pathlib import Path

import onnx
import torch
import torch.nn as nn
from omegaconf import DictConfig


def export_to_onnx(model: nn.Module, cfg: DictConfig) -> Path:
    model.eval()

    artifacts_dir = Path(cfg.data.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = artifacts_dir / f"model_{cfg.model.name}.onnx"

    image_size = tuple(cfg.data.image_size)
    dummy_input = torch.randn(1, 3, *image_size)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17,
        )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    print(f"[INFO] ONNX model saved and verified: {onnx_path}")
    return onnx_path
