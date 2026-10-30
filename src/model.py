"""
Neural Network Architectures for Plant Disease Classification:
1. Custom CNN baseline (Conv2D, BatchNorm, ReLU, MaxPooling, Dropout, Dense)
2. Transfer Learning using MobileNetV2 / ResNet50 with custom classification head
"""

import torch
import torch.nn as nn
from torchvision import models


class CustomCNN(nn.Module):
    """
    Custom Deep Convolutional Neural Network baseline.
    Architecture:
      - 4 Convolutional blocks with Conv2d, BatchNorm2d, ReLU, MaxPool2d, Dropout
      - Global Adaptive Average Pooling
      - Fully Connected Dense Classifier with BatchNorm1d and Dropout
    """
    def __init__(self, num_classes: int = 16, in_channels: int = 3):
        super(CustomCNN, self).__init__()

        # Block 1: 3 -> 32 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 -> 112
            nn.Dropout2d(0.20),
        )

        # Block 2: 32 -> 64 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56
            nn.Dropout2d(0.25),
        )

        # Block 3: 64 -> 128 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 -> 28
            nn.Dropout2d(0.30),
        )

        # Block 4: 128 -> 256 channels
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14
            nn.Dropout2d(0.35),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits

    def get_gradcam_target_layer(self) -> nn.Module:
        """Returns the final convolutional layer for Grad-CAM."""
        # Conv2d inside block4
        return self.block4[0]


class TransferLearningMobileNet(nn.Module):
    """
    Transfer Learning model using MobileNetV2 backbone.
    Lightweight, fast, and highly accurate for mobile/edge agriculture diagnostics.
    """
    def __init__(self, num_classes: int = 16, pretrained: bool = True, freeze_backbone: bool = False):
        super(TransferLearningMobileNet, self).__init__()

        weights = None
        if pretrained:
            try:
                weights = models.MobileNet_V2_Weights.DEFAULT
                self.backbone = models.mobilenet_v2(weights=weights)
            except Exception as e:
                print(f"[!] Pretrained weights download skipped ({e}). Using initialized backbone.")
                self.backbone = models.mobilenet_v2(weights=None)
        else:
            self.backbone = models.mobilenet_v2(weights=None)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self) -> nn.Module:
        """Returns the final convolutional block of MobileNetV2."""
        return self.backbone.features[-1]


class TransferLearningResNet(nn.Module):
    """
    Transfer Learning model using ResNet50 backbone.
    Deep residual architecture for complex multi-class plant pathology.
    """
    def __init__(self, num_classes: int = 16, pretrained: bool = True, freeze_backbone: bool = False):
        super(TransferLearningResNet, self).__init__()

        weights = None
        if pretrained:
            try:
                weights = models.ResNet50_Weights.DEFAULT
                self.backbone = models.resnet50(weights=weights)
            except Exception as e:
                print(f"[!] Pretrained weights download skipped ({e}). Using initialized backbone.")
                self.backbone = models.resnet50(weights=None)
        else:
            self.backbone = models.resnet50(weights=None)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.40),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self) -> nn.Module:
        """Returns the final residual block of ResNet50."""
        return self.backbone.layer4[-1]


def build_model(
    model_name: str = "custom_cnn",
    num_classes: int = 16,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Factory method to construct model by name.
    Options: 'custom_cnn', 'mobilenet_v2', 'resnet50'
    """
    name = model_name.lower().replace("-", "_")
    if name == "custom_cnn":
        return CustomCNN(num_classes=num_classes)
    elif name in ["mobilenet_v2", "mobilenet"]:
        return TransferLearningMobileNet(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif name in ["resnet50", "resnet"]:
        return TransferLearningResNet(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from 'custom_cnn', 'mobilenet_v2', 'resnet50'.")
