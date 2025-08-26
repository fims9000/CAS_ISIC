from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import cv2
import torch
from torchvision import transforms
import PIL.Image as Image


@dataclass
class XAIResult:
    """Результат XAI-анализа одного изображения.

    Пока содержит только заглушки под карты важности и метаданные.
    """

    heatmap: Optional[np.ndarray] = None
    overlay: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None


class BaseExplainer:
    """Базовый класс для объяснителей.

    Конкретные реализации должны переопределить метод ``explain``.
    """

    def __init__(self, model: Any, device: Any) -> None:
        self.model = model
        self.device = device

    def explain(self, image_path: Path, output_dir: Path) -> XAIResult:
        """Сгенерировать объяснение для ``image_path`` и сохранить артефакты в ``output_dir``.

        Заглушка. Возвращает пустой ``XAIResult``.
        """
        _ = image_path, output_dir
        return XAIResult(meta={"status": "not_implemented"})


# ===== Общие утилиты XAI =====

def prepare_cls_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Не удалось загрузить изображение: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_input_tensor(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    img_pil = Image.fromarray(image_rgb)
    tfm = prepare_cls_transform()
    x = tfm(img_pil).unsqueeze(0).to(device)
    return x


def normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    if hm.size == 0:
        return hm
    hm = hm.astype(np.float32)
    hm -= hm.min()
    denom = float(hm.max()) if float(hm.max()) > 1e-8 else 1.0
    hm /= denom
    return np.clip(hm, 0.0, 1.0)


def overlay_heatmap_on_image(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = (alpha * colored + (1 - alpha) * image_rgb.astype(np.float32)).astype(np.uint8)
    return overlay


def simple_gradient_heatmap(model: Any, x: torch.Tensor, target: int) -> np.ndarray:
    """Простейшая saliency-карта: |d logit[target] / d input|, усреднённая по каналам."""
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    logit = logits[0, target]
    logit.backward(retain_graph=False)
    grad = x.grad.detach().cpu().numpy()[0]
    grad_abs = np.abs(grad).mean(axis=0)
    return normalize_heatmap(grad_abs)


