import lightning as L
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchmetrics import Accuracy, F1Score, Precision, Recall

from leaf_classifier.models.baseline import BaselineCNN
from leaf_classifier.models.cnn import PlantDiseaseCNN


def build_model(cfg: DictConfig, num_classes: int) -> nn.Module:
    if cfg.model.name == "baseline":
        return BaselineCNN(num_classes=num_classes, dropout_rate=cfg.model.dropout_rate)
    return PlantDiseaseCNN(num_classes=num_classes, dropout_rate=cfg.model.dropout_rate)


class PlantDiseaseModule(L.LightningModule):
    def __init__(self, cfg: DictConfig, num_classes: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.model = build_model(cfg, num_classes)
        self.criterion = nn.CrossEntropyLoss()

        metrics_kwargs = {"task": "multiclass", "num_classes": num_classes}
        self.train_acc = Accuracy(**metrics_kwargs)
        self.val_acc = Accuracy(**metrics_kwargs)
        self.val_f1 = F1Score(**metrics_kwargs, average="macro")
        self.val_precision = Precision(**metrics_kwargs, average="macro")
        self.val_recall = Recall(**metrics_kwargs, average="macro")
        self.val_top3_acc = Accuracy(**metrics_kwargs, top_k=3)
        self.test_acc = Accuracy(**metrics_kwargs)
        self.test_f1 = F1Score(**metrics_kwargs, average="macro")
        self.test_precision = Precision(**metrics_kwargs, average="macro")
        self.test_recall = Recall(**metrics_kwargs, average="macro")
        self.test_top3_acc = Accuracy(**metrics_kwargs, top_k=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        self.train_acc.update(preds, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        probs = logits.softmax(dim=1)

        self.val_acc.update(preds, labels)
        self.val_f1.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_top3_acc.update(probs, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val_top3_acc", self.val_top3_acc, on_step=False, on_epoch=True)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        probs = logits.softmax(dim=1)

        self.test_acc.update(preds, labels)
        self.test_f1.update(preds, labels)
        self.test_precision.update(preds, labels)
        self.test_recall.update(preds, labels)
        self.test_top3_acc.update(probs, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True)
        self.log("test_top3_acc", self.test_top3_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.training.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.cfg.training.scheduler.factor,
            patience=self.cfg.training.scheduler.patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
