from pathlib import Path
from typing import Any

import numpy as np
import torch
from captum.attr import IntegratedGradients

from .base import BaseExplainer, XAIResult, load_rgb, to_input_tensor, normalize_heatmap, overlay_heatmap_on_image


class IntegratedGradientsExplainer(BaseExplainer):
    """Заготовка объяснителя Integrated Gradients.

    Реализацию градиентной интеграции добавим позже.
    """

    def __init__(self, model: Any, device: Any) -> None:
        super().__init__(model, device)

    def explain(self, image_path: Path, output_dir: Path) -> XAIResult:
        image_rgb = load_rgb(image_path)
        x = to_input_tensor(image_rgb, self.device)
        self.model.eval()

        # Целевая метка — максимум по логитам
        with torch.no_grad():
            logits = self.model(x)
            target = int(torch.argmax(logits, dim=1).item())
        # Лёгкий sanity-log (в stdout)
        try:
            print(f"[XAI][IG] target={target} logits_shape={tuple(logits.shape)}")
        except Exception:
            pass

        ig = IntegratedGradients(self.model)
        attributions, _ = ig.attribute(x, target=target, return_convergence_delta=True)
        # Суммируем по каналам и нормализуем
        attr = attributions.squeeze(0).detach().cpu().numpy()
        attr = np.abs(attr).sum(axis=0)
        heat = normalize_heatmap(attr)
        overlay = overlay_heatmap_on_image(image_rgb, heat)
        return XAIResult(heatmap=heat, overlay=overlay, meta={"method": "integrated_gradients", "target": target})


