from pathlib import Path

import fire
from hydra import compose, initialize_config_dir


def _load_config(overrides: list[str]):
    config_dir = str(Path(__file__).parent.resolve() / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name="config", overrides=overrides)


def train(*overrides: str) -> None:
    cfg = _load_config(list(overrides))
    from leaf_classifier.training.trainer import run_training

    run_training(cfg)


def infer(image_path: str, *overrides: str) -> dict:
    cfg = _load_config(list(overrides))
    from leaf_classifier.inference.predictor import run_inference

    return run_inference(cfg, Path(image_path))


if __name__ == "__main__":
    fire.Fire({"train": train, "infer": infer})
