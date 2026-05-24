import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction_ratio, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale.view(-1, scale.shape[1], 1, 1)


class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.block1 = self._make_block(3, 64)
        self.block2 = self._make_block(64, 128)
        self.block3 = self._make_block(128, 256)
        self.block4 = self._make_block(256, 512)
        self.block5 = self._make_block(512, 512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _make_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            SEBlock(out_channels),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool(x)
        return self.classifier(x)
