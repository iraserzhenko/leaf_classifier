from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from leaf_classifier.data.datamodule import PlantDiseaseDataModule
from leaf_classifier.training.module import PlantDiseaseModule
from leaf_classifier.utils.export import export_to_onnx
from leaf_classifier.utils.plots import TrainingPlotCallback


def _get_git_commit() -> str:
    try:
        import git

        repo = git.Repo(search_parent_directories=True)
        return repo.head.commit.hexsha[:8]
    except Exception:
        return "unknown"


def _pull_dvc_data(targets: list[str] | None = None) -> None:
    try:
        from dvc.repo import Repo

        with Repo() as repo:
            repo.pull(targets=targets)
        print("[INFO] DVC pull completed.")
    except Exception as exc:
        print(f"[WARN] DVC pull skipped: {exc}")


def run_training(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)

    _pull_dvc_data(targets=[cfg.data.data_dir])

    datamodule = PlantDiseaseDataModule(cfg)
    datamodule.setup()

    module = PlantDiseaseModule(cfg, num_classes=datamodule.num_classes)

    git_commit = _get_git_commit()
    logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.tracking_uri,
        tags={"git_commit": git_commit, "model": cfg.model.name},
    )
    logger.log_hyperparams(
        {
            "model": cfg.model.name,
            "dropout_rate": cfg.model.dropout_rate,
            "batch_size": cfg.training.batch_size,
            "epochs": cfg.training.epochs,
            "learning_rate": cfg.training.learning_rate,
            "image_size": cfg.data.image_size,
            "seed": cfg.seed,
            "git_commit": git_commit,
        }
    )

    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"{cfg.model.name}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=cfg.training.early_stopping_patience,
        min_delta=cfg.training.early_stopping_min_delta,
    )
    plot_callback = TrainingPlotCallback(plots_dir=Path("plots"))

    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        callbacks=[checkpoint_callback, early_stopping, plot_callback],
        logger=logger,
        accelerator="auto",
        devices="auto",
        deterministic=True,
    )

    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule, ckpt_path="best")

    best_ckpt = checkpoint_callback.best_model_path
    if best_ckpt:
        ckpt_data = torch.load(best_ckpt, map_location="cpu")
        inner_state = {
            key[len("model.") :]: value
            for key, value in ckpt_data["state_dict"].items()
            if key.startswith("model.")
        }
        module.model.load_state_dict(inner_state)

    export_to_onnx(module.model, cfg)
    print("[INFO] Training complete.")
