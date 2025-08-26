from pathlib import Path
from typing import Any, Dict, Optional, List

from .base import XAIResult
from .integrated_gradients import IntegratedGradientsExplainer
from .shap_explainer import SHAPExplainer
from .grad_cam import GradCAMExplainer
from .lime_explainer import LIMEExplainer


def run_xai(
    model: Any,
    device: Any,
    image_path: Path,
    output_dir: Path,
    method: str = "grad_cam",
    options: Optional[Dict[str, Any]] = None,
) -> XAIResult:
    """Запустить XAI-анализ выбранным методом.

    Пока реализация заглушечная: выбирает объяснитель и возвращает пустой ``XAIResult``.
    """
    _ = options
    explainer_map = {
        "integrated_gradients": IntegratedGradientsExplainer,
        "shap": SHAPExplainer,
        "grad_cam": GradCAMExplainer,
        "lime": LIMEExplainer,
    }
    method = method.lower()
    if method not in explainer_map:
        method = "grad_cam"
    explainer = explainer_map[method](model, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    return explainer.explain(image_path, output_dir)


def run_xai_all(
    model: Any,
    device: Any,
    image_path: Path,
    output_dir: Path,
    methods: Optional[List[str]] = None,
) -> Dict[str, XAIResult]:
    methods = methods or ["integrated_gradients", "shap", "grad_cam", "lime"]
    results: Dict[str, XAIResult] = {}
    for m in methods:
        try:
            res = run_xai(model, device, image_path, output_dir, method=m)
            results[m] = res
        except Exception as e:
            results[m] = XAIResult(meta={"method": m, "error": str(e)})
    return results


def build_xai_filename(stem: str, model_tag: str, method_key: str) -> str:
    """Возвращает имя файла XAI_<stem>_<model_tag>_(IG|GradCAM|SHAP|LIME).png"""
    mapping = {
        "integrated_gradients": "IG",
        "grad_cam": "GradCAM",
        "shap": "SHAP",
        "lime": "LIME",
    }
    suffix = mapping.get(method_key, method_key.upper())
    return f"XAI_{stem}_{model_tag}_{suffix}.png"


