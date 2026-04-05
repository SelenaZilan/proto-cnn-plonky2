import torch
from torch import nn
class SmallIrisCNN(nn.Module):
    """Small CNN for iris classification / feature extraction.

    Intended for experiments where a Plonky2-style proof over the full forward pass
    is conceivable: few layers, moderate channel counts, fixed 128-D embedding before ``fc``.

    Expects square RGB inputs (default ``input_size=64`` in ``train.py`` / eval). With 64x64
    and three stride-2 pools, the spatial grid before global pooling is 8x8x128.
    """

    def __init__(
        self,
        num_classes: int = 1500,
        embedding_dim: int = 128,
        c1: int = 32,
        c2: int = 64,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.c1 = c1
        self.c2 = c2
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, embedding_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def feature_extract_avg_pool(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extract_avg_pool(x)
        return self.fc(x)

    def config_dict(self):
        return {
            "num_classes": self.fc.out_features,
            "embedding_dim": self.embedding_dim,
            "c1": self.c1,
            "c2": self.c2,
        }
