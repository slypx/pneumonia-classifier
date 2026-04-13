import torch.nn as nn
import timm


class PneumoniaClassifier(nn.Module):
    """
    EfficientNet-B0 based classifier for pneumonia detection.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.4, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.num_features, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)