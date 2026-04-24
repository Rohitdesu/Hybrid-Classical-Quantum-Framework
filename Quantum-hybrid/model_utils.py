import base64
import io
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = {0: "Benign", 1: "Malignant"}


class QuantumInspiredProcessingLayer(nn.Module):
    def __init__(self, in_features: int = 2048, hidden_features: int = 512):
        super().__init__()
        self.map = nn.Linear(in_features, hidden_features)
        self.interact = nn.Linear(hidden_features, hidden_features)
        self.norm = nn.LayerNorm(hidden_features)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mapped = torch.tanh(self.map(x))
        interacted = torch.relu(self.interact(mapped))
        phase_encoded = torch.sin(interacted) + torch.cos(interacted)
        return self.dropout(self.norm(phase_encoded))


class FeatureSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = x.unsqueeze(1)
        attended, weights = self.attn(tokens, tokens, tokens, need_weights=True)
        output = self.norm(tokens + self.dropout(attended))
        return output.squeeze(1), weights


class HybridCancerClassifier(nn.Module):
    def __init__(self, dropout1: float = 0.35, dropout2: float = 0.30):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.quantum_layer = QuantumInspiredProcessingLayer(2048, 512)
        self.attention = FeatureSelfAttention(embed_dim=512, num_heads=8, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x).flatten(1)
        enhanced = self.quantum_layer(features)
        attended, attention_weights = self.attention(enhanced)
        logits = self.classifier(attended).squeeze(1)
        return logits, attention_weights


def build_inference_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_checkpoint_model(checkpoint_path: str | Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = HybridCancerClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device, checkpoint


def predict_pil_image(
    image: Image.Image,
    model: nn.Module,
    device: torch.device,
    transform: transforms.Compose,
) -> Dict[str, float | str]:
    rgb_image = image.convert("RGB")
    input_tensor = transform(rgb_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(input_tensor)
        malignant_probability = torch.sigmoid(logits).item()

    benign_probability = 1.0 - malignant_probability
    pred_index = 1 if malignant_probability >= 0.5 else 0
    confidence = malignant_probability if pred_index == 1 else benign_probability

    return {
        "predicted_label": CLASS_NAMES[pred_index],
        "confidence": confidence,
        "malignant_probability": malignant_probability,
        "benign_probability": benign_probability,
    }


def image_to_data_url(image: Image.Image, max_size: int = 700) -> str:
    preview = image.copy()
    preview.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    preview.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
