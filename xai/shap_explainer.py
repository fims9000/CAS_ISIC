from pathlib import Path
from typing import Any

import numpy as np
import torch
from captum.attr import GradientShap

from .base import BaseExplainer, XAIResult, load_rgb, to_input_tensor, normalize_heatmap, overlay_heatmap_on_image, simple_gradient_heatmap


class SHAPExplainer(BaseExplainer):
    """Объяснитель SHAP (GradientSHAP из Captum) для стабильных карт.

    GradientSHAP даёт SHAP-подобные атрибуции и работает устойчиво на CNN.
    """

    def __init__(self, model: Any, device: Any) -> None:
        super().__init__(model, device)

    def explain(self, image_path: Path, output_dir: Path) -> XAIResult:
        image_rgb = load_rgb(image_path)
        x = to_input_tensor(image_rgb, self.device)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(x)
            target = int(torch.argmax(logits, dim=1).item())

        try:
            gs = GradientShap(self.model)
            # Бейзлайны: нулевое изображение и слегка зашумлённый ноль
            baseline_zero = torch.zeros_like(x)
            noise = torch.randn_like(x) * 0.02
            baselines = (baseline_zero, baseline_zero + noise)
            attributions = gs.attribute(x, baselines=baselines, target=target, n_samples=32, stdevs=0.01)
            attr = attributions.squeeze(0).detach().cpu().numpy()  # CxHxW
            heat = normalize_heatmap(np.abs(attr).sum(axis=0))
            overlay = overlay_heatmap_on_image(image_rgb, heat)
        except Exception:
            # Фоллбек: простая градиентная карта
            heat = simple_gradient_heatmap(self.model, x, target)
            overlay = overlay_heatmap_on_image(image_rgb, heat)
        return XAIResult(heatmap=heat, overlay=overlay, meta={"method": "shap", "target": target})


