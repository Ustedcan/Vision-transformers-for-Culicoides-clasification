import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

# ---------------------------- Model ----------------------------
class ViTFeatureExtractor(nn.Module):
    """ViT-B/16 backbone returning the CLS embedding (dim=768)."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = vit_b_16(weights=weights)
        # Remove classification head to expose embeddings
        self.model.heads = nn.Identity()

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # [B, 768]

class MLPHead(nn.Module):
    def __init__(self, in_dim: int = 768, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ViTEndToEnd(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = ViTFeatureExtractor(pretrained=pretrained, freeze_backbone=freeze_backbone)
        self.head = MLPHead(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.head(z)


