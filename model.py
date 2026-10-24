import io
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms


class BottleneckDec(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck_ratio=4):
        super().__init__()
        mid = max(out_ch // bottleneck_ratio, 8)
        self.conv1 = nn.Conv2d(in_ch, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        return self.act(self.bn3(self.conv3(out)) + identity)


class ResNet50DepthSwAV(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.start = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.dec4 = BottleneckDec(2048, 512)
        self.dec3 = BottleneckDec(512 + 1024, 256)
        self.dec2 = BottleneckDec(256 + 512, 128)
        self.dec1 = BottleneckDec(128 + 256, 64)
        self.dec0 = BottleneckDec(64 + 64, 64, bottleneck_ratio=2)
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x0 = self.start(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        d4 = self.dec4(x4)
        u3 = F.interpolate(d4, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, x3], dim=1))
        u2 = F.interpolate(d3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, x2], dim=1))
        u1 = F.interpolate(d2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, x1], dim=1))
        d0 = self.dec0(torch.cat([d1, x0], dim=1))
        out = F.interpolate(d0, size=size, mode="bilinear", align_corners=False)
        return F.softplus(self.head(out))


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


@lru_cache(maxsize=1)
def load_model(weights_path):
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model = ResNet50DepthSwAV()
    model.load_state_dict(state)
    return model.eval()


@torch.inference_mode()
def infer_depth_png_and_npy(model, image_bytes):
    with Image.open(io.BytesIO(image_bytes)) as image:
        tensor = IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0)
    prediction = model(tensor)[0, 0].cpu().numpy().astype(np.float32)
    if not np.isfinite(prediction).all():
        raise ValueError("Модель вернула некорректную карту глубины")
    npy = io.BytesIO()
    np.save(npy, prediction, allow_pickle=False)
    minimum, maximum = float(prediction.min()), float(prediction.max())
    scale = maximum - minimum
    normalized = (prediction - minimum) / (scale if scale > 1e-6 else 1)
    pixels = (normalized * 255).astype(np.uint8)
    png = io.BytesIO()
    Image.fromarray(pixels).save(png, format="PNG")
    return png.getvalue(), npy.getvalue()


def predict(image_bytes, weights_path):
    model = load_model(str(Path(weights_path).resolve()))
    return infer_depth_png_and_npy(model, image_bytes)
