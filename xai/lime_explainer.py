from pathlib import Path
from typing import Any

import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries

from .base import BaseExplainer, XAIResult, load_rgb


class LIMEExplainer(BaseExplainer):
    """Объяснитель LIME (через lime-image)."""

    def __init__(self, model: Any, device: Any) -> None:
        super().__init__(model, device)

    def explain(self, image_path: Path, output_dir: Path) -> XAIResult:
        image_rgb = load_rgb(image_path)
        explainer = lime_image.LimeImageExplainer()

        def predict_fn(images_np):
            # images_np: N x H x W x 3 в [0,255]; нормализуем и кидаем в модель через базовый tfm
            import torch
            from .base import to_input_tensor
            outs = []
            for im in images_np:
                x = to_input_tensor(im.astype(np.uint8), self.device)
                with torch.no_grad():
                    logits = self.model(x)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                outs.append(probs)
            return np.array(outs)

        explanation = explainer.explain_instance(
            image_rgb,
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=1000,
        )
        try:
            print("[XAI][LIME] explanation computed")
        except Exception:
            pass
        # Берём топ-метку
        label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            label,
            positive_only=True,
            num_features=5,
            hide_rest=False,
        )
        overlay = (mark_boundaries(temp / 255.0, mask) * 255).astype(np.uint8)
        heat = mask.astype(np.float32)
        return XAIResult(heatmap=heat, overlay=overlay, meta={"method": "lime", "target": int(label)})


