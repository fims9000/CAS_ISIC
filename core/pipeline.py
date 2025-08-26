import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import gc
import numpy as np
import torch
from torchvision import models, transforms

# Локальные импорты сегментации
from classification_and_segmentation.test_generation_segm import (
    load_model as load_unetpp,
    infer_one as infer_segmentation_one,
    build_device as build_seg_device,
)
from xai.runner import run_xai_all, build_xai_filename


CLASSES: List[str] = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']


@dataclass
class PipelineConfig:
    seg_checkpoint: Path
    cls_checkpoint: Path
    output_dir: Path
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    # Параметры сегментации (совместимые с test_generation_segm.py)
    input_size: int = 0
    threshold: float = 0.8
    use_clahe: bool = False
    gamma: float = 1.0
    unsharp: float = 0.0
    tta: bool = True
    adaptive: bool = False
    morph: int = 1
    min_area: int = 500
    fill_holes: bool = False
    keep_largest: bool = True
    debug: bool = True
    # Авто-ресайз для стабильности: если input_size==0, ограничиваем максимум
    auto_resize: bool = True
    max_side: int = 512
    seg_arch: str = 'old'
    clear_border: bool = True
    heuristics: bool = True
    # XAI
    xai_enabled: bool = False
    xai_methods: List[str] = field(default_factory=lambda: [
        'integrated_gradients', 'shap', 'grad_cam', 'lime'
    ])
    xai_dir_name: str = 'XAI_results'


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_write_png(path: Path, img: np.ndarray) -> None:
    try:
        arr = img
        if arr is None:
            return
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = (np.stack([arr, arr, arr], axis=-1))
        if arr.dtype != np.uint8:
            # нормализуем в [0,255]
            mn = float(arr.min()) if arr.size else 0.0
            mx = float(arr.max()) if arr.size else 1.0
            denom = (mx - mn) if (mx - mn) > 1e-8 else 1.0
            arr = ((arr - mn) / denom * 255.0).clip(0, 255).astype(np.uint8)
        # гарантируем RGB->BGR для imwrite
        if arr.shape[-1] == 3:
            arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            arr_bgr = arr
        cv2.imwrite(str(path), arr_bgr)
    except Exception:
        pass


def _load_resnet18_classifier(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    num_classes = len(CLASSES)
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(str(checkpoint), map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    # Проверка совместимости головы
    if 'fc.weight' in state:
        out_features = state['fc.weight'].shape[0]
        if out_features != num_classes:
            raise RuntimeError(
                f"Несовместимый чекпойнт классификатора: fc.out_features={out_features}, ожидается {num_classes}. "
                f"Актуальные классы: {CLASSES}"
            )
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def _softmax_to_onehot(probs: np.ndarray) -> List[float]:
    idx = int(np.argmax(probs))
    onehot = [0.0] * len(CLASSES)
    onehot[idx] = 1.0
    return onehot


def _prepare_cls_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Не удалось загрузить изображение: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _infer_class(model: torch.nn.Module, image_rgb: np.ndarray, device: torch.device) -> Tuple[int, List[float]]:
    transform = _prepare_cls_transform()
    pil_like = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # для ToTensor не важно, но оставим RGB
    # Преобразуем через PIL-пайплайн: используем cv2 -> PIL совместимо через numpy
    import PIL.Image as Image
    img_pil = Image.fromarray(image_rgb)
    x = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    cls_idx = int(np.argmax(probs))
    return cls_idx, probs.tolist()


def _write_csv_header(csv_path: Path) -> None:
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image'] + CLASSES)


def _append_csv_row(csv_path: Path, image_stem: str, onehot: List[float]) -> None:
    with csv_path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([image_stem] + [f"{v:.1f}" for v in onehot])


class CASPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = build_seg_device(config.device)
        # Модели
        # Выбираем стартовую архитектуру
        start_arch = 'new' if config.seg_arch != 'old' else 'old'
        self._current_arch = start_arch
        self.seg_model = load_unetpp(config.seg_checkpoint, self.device, arch=start_arch, debug=False)
        self.cls_model = _load_resnet18_classifier(config.cls_checkpoint, self.device)

        # Выходные пути
        self.output_dir = config.output_dir
        _ensure_dir(self.output_dir)
        # Подготовка директории XAI (если режим включен)
        if self.config.xai_enabled:
            try:
                _ensure_dir(self.output_dir / self.config.xai_dir_name)
            except Exception:
                pass

        # CSV для предсказаний классов (в директории вывода)
        self.csv_path = self.output_dir / 'predictions.csv'
        _write_csv_header(self.csv_path)

        # Быстрая проверка совместимости сегментационной модели: один выходной канал
        with torch.no_grad():
            test_tensor = torch.zeros(1, 3, max(64, self.config.input_size), max(64, self.config.input_size), device=self.device)
            out = self.seg_model(test_tensor)
            if out.ndim != 4 or out.shape[1] != 1:
                raise RuntimeError(
                    f"Несовместимый сегментационный чекпойнт/модель: ожидается 1 выходной канал, получено shape={tuple(out.shape)}"
                )
        # Очистка после тестового прогона
        self._cleanup()

    def _cleanup(self) -> None:
        try:
            if isinstance(self.device, torch.device) and self.device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass

    def _decide_input_size(self, image_path: Path) -> int:
        # Если указали фиксированный размер — используем его
        if self.config.input_size and self.config.input_size > 0:
            return int(self.config.input_size)
        if not self.config.auto_resize:
            return 0
        # Логика авто-ограничения: если сторона больше max_side — используем max_side, иначе 0 (оригинал)
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                return self.config.max_side
            h, w = img.shape[:2]
            if max(h, w) > int(self.config.max_side):
                return int(self.config.max_side)
            return 0
        except Exception:
            return int(self.config.max_side)

    @staticmethod
    def _white_ratio(mask: np.ndarray) -> float:
        h, w = mask.shape[:2]
        if h * w == 0:
            return 0.0
        return float(np.count_nonzero(mask)) / float(h * w)

    def _remove_border_components(self, mask: np.ndarray) -> np.ndarray:
        try:
            m = (mask > 0).astype(np.uint8)
            h, w = m.shape[:2]
            num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            keep = np.zeros_like(m)
            for i in range(1, num):
                x, y, ww, hh, area = stats[i]
                touches = (x == 0) or (y == 0) or (x + ww >= w) or (y + hh >= h)
                if not touches:
                    keep[labels == i] = 1
            return (keep * 255).astype(np.uint8)
        except Exception:
            return mask

    def _heuristic_adjust(self, mask: np.ndarray) -> np.ndarray:
        if not self.config.heuristics:
            return mask
        try:
            ratio = self._white_ratio(mask)
            kernel = np.ones((3, 3), np.uint8)
            out = mask.copy()
            if ratio > 0.9:
                out = cv2.erode(out, kernel, iterations=2)
            elif ratio > 0.7:
                out = cv2.erode(out, kernel, iterations=1)
            elif ratio < 0.002:
                out = cv2.dilate(out, kernel, iterations=1)
            return out
        except Exception:
            return mask

    def process_image(self, image_path: Path) -> Tuple[Path, int]:
        # 1) Сегментация с предобработкой (по правилам test_generation_segm)
        # Выбираем безопасный input_size, чтобы избежать OOM на очень больших изображениях
        safe_input_size = self._decide_input_size(image_path)
        img_rgb, mask = infer_segmentation_one(
            self.seg_model, image_path, safe_input_size, self.config.threshold, self.device,
            debug=self.config.debug, use_clahe=self.config.use_clahe, gamma=self.config.gamma, unsharp=self.config.unsharp,
            tta=self.config.tta, adaptive=self.config.adaptive, morph=self.config.morph,
            min_area=self.config.min_area, fill_holes=self.config.fill_holes, keep_largest=self.config.keep_largest
        )

        # Если маска пустая и включён auto-режим — попробуем альтернативную архитектуру
        if self.config.seg_arch == 'auto':
            white_ratio = float(np.count_nonzero(mask)) / float(mask.size) if mask.size > 0 else 0.0
            if white_ratio == 0.0 and self._current_arch == 'new':
                # Перезагружаем модель в старой архитектуре и повторяем инференс
                self.seg_model = load_unetpp(self.config.seg_checkpoint, self.device, arch='old', debug=False)
                self._current_arch = 'old'
                img_rgb, mask = infer_segmentation_one(
                    self.seg_model, image_path, safe_input_size, self.config.threshold, self.device,
                    debug=self.config.debug, use_clahe=self.config.use_clahe, gamma=self.config.gamma, unsharp=self.config.unsharp,
                    tta=self.config.tta, adaptive=self.config.adaptive, morph=self.config.morph,
                    min_area=self.config.min_area, fill_holes=self.config.fill_holes, keep_largest=self.config.keep_largest
                )

        # Доп. постобработка
        if self.config.clear_border:
            mask = self._remove_border_components(mask)
        mask = self._heuristic_adjust(mask)

        # Сохраняем маску в директории вывода
        stem = image_path.stem
        # Сохраняем в подпапку masks внутри директории вывода
        out_mask = (self.output_dir / 'masks') / f"{stem}_mask.png"
        _ensure_dir(out_mask.parent)
        cv2.imwrite(str(out_mask), mask)

        # 2) Классификация на исходном изображении (без предобработки сегментации)
        orig_rgb = _read_rgb(image_path)
        cls_idx, probs = _infer_class(self.cls_model, orig_rgb, self.device)
        onehot = _softmax_to_onehot(np.array(probs))
        _append_csv_row(self.csv_path, stem, onehot)

        # 2.1) XAI (если режим включен) — генерируем карты методами и сохраняем
        if self.config.xai_enabled:
            try:
                xai_root = self.output_dir / self.config.xai_dir_name
                _ensure_dir(xai_root)
                # Подпапка с именем изображения (stem)
                xai_subdir = xai_root / f"{stem}"
                _ensure_dir(xai_subdir)
                # Очистим старые артефакты, чтобы не плодить дубликаты
                try:
                    for fp in xai_subdir.glob('*'):
                        try:
                            if fp.is_file():
                                fp.unlink()
                        except Exception:
                            pass
                except Exception:
                    pass
                # Генерация XAI для классификации
                print(f"[XAI] start CLS for {stem} methods={self.config.xai_methods}")
                cls_results = run_xai_all(self.cls_model, self.device, image_path, xai_subdir, methods=self.config.xai_methods)
                for key, res in cls_results.items():
                    file_name = build_xai_filename(stem, 'cls', key)
                    if res and getattr(res, 'overlay', None) is not None:
                        _safe_write_png(xai_subdir / file_name, res.overlay)
                    elif res and getattr(res, 'heatmap', None) is not None:
                        _safe_write_png(xai_subdir / file_name, res.heatmap)
                # Генерация XAI для сегментации: используем обёртку в скалярный логит
                print(f"[XAI] start SEGM for {stem} methods={self.config.xai_methods}")
                segm_wrapped = SegmScalarWrapper(self.seg_model).to(self.device)
                segm_wrapped.eval()
                segm_results = run_xai_all(segm_wrapped, self.device, image_path, xai_subdir, methods=self.config.xai_methods)
                # Подсказка: если сегментационная модель имеет 1-канальный выход, Grad-CAM/SHAP могут быть менее информативны,
                # но фоллбеки обеспечат сохранение карт.
                for key, res in segm_results.items():
                    file_name = build_xai_filename(stem, 'segm', key)
                    if res and getattr(res, 'overlay', None) is not None:
                        _safe_write_png(xai_subdir / file_name, res.overlay)
                    elif res and getattr(res, 'heatmap', None) is not None:
                        _safe_write_png(xai_subdir / file_name, res.heatmap)
            except Exception:
                import traceback as _tb
                print("[XAI] failed:\n" + ''.join(_tb.format_exc()))

        # Явная очистка крупных объектов
        try:
            del img_rgb
            del mask
            del orig_rgb
        except Exception:
            pass
        self._cleanup()

        return out_mask, cls_idx

    def process_directory(self, input_dir: Path) -> int:
        # Ищем изображения рекурсивно
        exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        files: List[Path] = []
        for e in exts:
            files.extend(Path(input_dir).rglob(e))
        files = sorted(files)
        count = 0
        for img_path in files:
            try:
                self.process_image(img_path)
                count += 1
            except Exception as e:
                # Простая диагностика
                print(f"Ошибка обработки {img_path.name}: {e}")
        return count


# ===== Вспомогательная обёртка для сегментационной модели =====
class SegmScalarWrapper(torch.nn.Module):
    """Оборачивает сегментационную модель в классификационную форму.

    Возвращает один логит на изображение: среднее значение выхода модели.
    Это позволяет использовать общие XAI-методы (IG/SHAP/Grad-CAM/LIME).
    """

    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base_model(x)
        # y: (N, C, H, W) или (N, 1, H, W) или иной; приводим к одному числу на батч
        if y.ndim >= 2:
            # усредняем по всем измерениям, кроме батча
            dims = tuple(range(1, y.ndim))
            out = y.float().mean(dim=dims, keepdim=False)
        else:
            out = y.float()
        # Приводим к форме (N, 1) как у логитов для классификации
        if out.ndim == 1:
            out = out.unsqueeze(1)
        return out


