from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback


class TrainingPlotCallback(Callback):
    def __init__(self, plots_dir: Path) -> None:
        super().__init__()
        self.plots_dir = plots_dir
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_accs: list[float] = []
        self.val_f1s: list[float] = []
        self.val_precisions: list[float] = []
        self.val_recalls: list[float] = []

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        metrics = trainer.callback_metrics
        if "train_loss_epoch" in metrics:
            self.train_losses.append(float(metrics["train_loss_epoch"]))

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        metrics = trainer.callback_metrics
        if "val_loss" in metrics:
            self.val_losses.append(float(metrics["val_loss"]))
        if "val_acc" in metrics:
            self.val_accs.append(float(metrics["val_acc"]))
        if "val_f1" in metrics:
            self.val_f1s.append(float(metrics["val_f1"]))
        if "val_precision" in metrics:
            self.val_precisions.append(float(metrics["val_precision"]))
        if "val_recall" in metrics:
            self.val_recalls.append(float(metrics["val_recall"]))

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        save_training_plots(
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            val_accs=self.val_accs,
            val_f1s=self.val_f1s,
            val_precisions=self.val_precisions,
            val_recalls=self.val_recalls,
            plots_dir=self.plots_dir,
        )


def save_training_plots(
    train_losses: list[float],
    val_losses: list[float],
    val_accs: list[float],
    val_f1s: list[float],
    val_precisions: list[float],
    val_recalls: list[float],
    plots_dir: Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    _save_loss_plot(train_losses, val_losses, plots_dir / "training_loss.png")
    _save_accuracy_plot(val_accs, plots_dir / "validation_accuracy.png")
    _save_metrics_plot(val_f1s, val_precisions, val_recalls, plots_dir / "validation_metrics.png")


def _save_loss_plot(train_losses: list[float], val_losses: list[float], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    if train_losses:
        ax.plot(epochs, train_losses, label="Train Loss", marker="o", markersize=3)
    val_epochs = range(1, len(val_losses) + 1)
    if val_losses:
        ax.plot(val_epochs, val_losses, label="Val Loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _save_accuracy_plot(val_accs: list[float], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if val_accs:
        epochs = range(1, len(val_accs) + 1)
        ax.plot(epochs, val_accs, label="Val Accuracy", color="green", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _save_metrics_plot(
    val_f1s: list[float],
    val_precisions: list[float],
    val_recalls: list[float],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if val_f1s:
        epochs = range(1, len(val_f1s) + 1)
        ax.plot(epochs, val_f1s, label="Val F1 (macro)", marker="o", markersize=3)
    if val_precisions:
        epochs = range(1, len(val_precisions) + 1)
        ax.plot(epochs, val_precisions, label="Val Precision (macro)", marker="s", markersize=3)
    if val_recalls:
        epochs = range(1, len(val_recalls) + 1)
        ax.plot(epochs, val_recalls, label="Val Recall (macro)", marker="^", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation Metrics (macro-averaged)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
