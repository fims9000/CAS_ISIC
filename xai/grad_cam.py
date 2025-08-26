from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
try:
    # Патчим проблемный __del__ в BaseCAM, чтобы не падал при отсутствии атрибута
    from pytorch_grad_cam.base_cam import BaseCAM as _BaseCAM
    if not hasattr(_BaseCAM, "__del__patched__"):
        def _safe_del(self):
            try:
                ag = getattr(self, "activations_and_grads", None)
                if ag is not None and hasattr(ag, "release"):
                    ag.release()
            except Exception:
                pass
        _BaseCAM.__del__ = _safe_del
        _BaseCAM.__del__patched__ = True
except Exception:
    pass
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from .base import BaseExplainer, XAIResult, load_rgb, to_input_tensor, normalize_heatmap, overlay_heatmap_on_image, simple_gradient_heatmap


class GradCAMExplainer(BaseExplainer):
    """Заготовка объяснителя Grad-CAM."""

    def __init__(self, model: Any, device: Any) -> None:
        super().__init__(model, device)

    def explain(self, image_path: Path, output_dir: Path) -> XAIResult:
        image_rgb = load_rgb(image_path)
        x = to_input_tensor(image_rgb, self.device)
        self.model.eval()

        # Выбираем последний conv-слой автоматически (для ResNet18 — layer4[-1])
        target_layers = []
        try:
            target_layers = [self.model.layer4[-1]]
        except Exception:
            # fallback: ищем последний сверточный слой рекурсивно
            target_layers = []
            for m in reversed(list(self.model.modules())):
                import torch.nn as nn
                if isinstance(m, nn.Conv2d):
                    target_layers = [m]
                    break
            if not target_layers:
                last_layer = list(self.model.children())[-1]
                target_layers = [last_layer]

        with torch.no_grad():
            logits = self.model(x)
            target = int(torch.argmax(logits, dim=1).item())
        try:
            print(f"[XAI][GradCAM] target={target}")
        except Exception:
            pass

        use_cuda = False
        try:
            import torch as _t
            use_cuda = isinstance(self.device, _t.device) and self.device.type == 'cuda'
        except Exception:
            use_cuda = False

        try:
            # Используем контекстный менеджер для корректного освобождения ресурсов
            cam_obj = None
            try:
                with GradCAM(model=self.model, target_layers=target_layers, use_cuda=use_cuda) as cam:
                    cam_obj = cam
                    grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(target)])
            finally:
                # Подстраховка от ошибки в __del__ библиотеки: гарантируем наличие поля
                try:
                    if cam_obj is not None and not hasattr(cam_obj, 'activations_and_grads'):
                        class _Dummy:
                            def release(self):
                                return None
                        cam_obj.activations_and_grads = _Dummy()
                except Exception:
                    pass
            heat = grayscale_cam[0]
            heat = normalize_heatmap(heat)
            # show_cam_on_image ждёт float в [0,1]
            img_float = (image_rgb.astype(np.float32) / 255.0)
            overlay = show_cam_on_image(img_float, heat, use_rgb=True)
            overlay = (overlay * 255.0).astype(np.uint8)
        except Exception:
            # Фоллбек: простая градиентная карта, чтобы всегда вернуть результат
            heat = simple_gradient_heatmap(self.model, x, target)
            overlay = overlay_heatmap_on_image(image_rgb, heat)
        return XAIResult(heatmap=heat, overlay=overlay, meta={"method": "grad_cam", "target": target})


