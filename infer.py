import json
from pathlib import Path

import fire
from hydra import compose, initialize_config_dir


def infer(image_path: str, *overrides: str) -> None:
    config_dir = str(Path(__file__).parent.resolve() / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=list(overrides))

    from leaf_classifier.inference.predictor import run_inference

    result = run_inference(cfg, Path(image_path))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    fire.Fire(infer)
