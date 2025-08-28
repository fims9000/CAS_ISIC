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
# Локальные функции для сегментации (отвязано от classification_and_segmentation)
def build_seg_device(option: str) -> torch.device:
    if option == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(option)


def _apply_preprocessing(img_rgb: np.ndarray, use_clahe: bool, gamma: float, unsharp: float) -> np.ndarray:
    out = img_rgb.copy()
    if use_clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if gamma and abs(gamma - 1.0) > 1e-3:
        g = max(0.2, min(3.0, float(gamma)))
        lut = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)]).astype("uint8")
        out = cv2.LUT(out, lut)
    if unsharp and unsharp > 0.0:
        sigma = 1.0
        blurred = cv2.GaussianBlur(out, (0, 0), sigma)
        out = cv2.addWeighted(out, 1.0 + float(unsharp), blurred, -float(unsharp), 0)
    return out


def _to_model_input(img_rgb: np.ndarray, input_size: int, device: torch.device) -> torch.Tensor:
    if input_size > 0:
        img_rgb = cv2.resize(img_rgb, (input_size, input_size), interpolation=cv2.INTER_LANCZOS4)
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def _postprocess_mask(mask: np.ndarray, morph: int, min_area: int, fill_holes: bool = False, keep_largest: bool = False) -> np.ndarray:
    out = mask.copy()
    if morph and morph >= 3:
        k = int(morph)
        k = k if k % 2 == 1 else k + 1
        kernel = np.ones((k, k), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    if fill_holes:
        m = (out > 0).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None and len(contours) > 0:
            filled = m.copy()
            hier = hierarchy[0]
            for i in range(len(contours)):
                parent = hier[i][3]
                if parent != -1:
                    cv2.drawContours(filled, contours, i, 255, thickness=cv2.FILLED)
            out = filled
    if min_area and min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), connectivity=8)
        keep = np.zeros_like(out)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                keep[labels == i] = 255
        out = keep
    if keep_largest:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), connectivity=8)
        if num_labels > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            out = (labels == largest).astype(np.uint8) * 255
    return out


def _binarize_probs(probs: torch.Tensor, threshold: float, adaptive: bool, out_h: int, out_w: int) -> np.ndarray:
    mask = (probs > threshold).float()
    mask_np = mask.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    if adaptive:
        prob_np = probs.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        if prob_np.shape != (out_h, out_w):
            prob_np = cv2.resize(prob_np, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        prob_u8 = np.clip(prob_np * 255.0, 0, 255).astype(np.uint8)
        _, mask_np = cv2.threshold(prob_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask_np = (mask_np > 0).astype(np.uint8)
    if mask_np.shape != (out_h, out_w):
        mask_np = cv2.resize(mask_np, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return (mask_np * 255).astype(np.uint8)


def _resize_probs(probs: torch.Tensor, out_h: int, out_w: int) -> np.ndarray:
    p = probs.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    if p.shape != (out_h, out_w):
        p = cv2.resize(p, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(p, 0.0, 1.0)


def _white_ratio(mask: np.ndarray) -> float:
    h, w = mask.shape[:2]
    if h * w == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(h * w)


def _logits_to_mask(logits: torch.Tensor, out_h: int, out_w: int, threshold: float, adaptive: bool, morph: int, min_area: int, fill_holes: bool, keep_largest: bool, debug: bool) -> np.ndarray:
    probs = torch.sigmoid(logits)
    mask_np = _binarize_probs(probs, threshold, adaptive, out_h, out_w)
    mask_np = _postprocess_mask(mask_np, morph, min_area, fill_holes=fill_holes, keep_largest=keep_largest)
    if debug:
        p = _resize_probs(probs, out_h, out_w)
        print(f"[DEBUG] probs min/max/mean: {float(p.min()):.4f}/{float(p.max()):.4f}/{float(p.mean()):.4f}")
    if _white_ratio(mask_np) >= 0.98:
        p = _resize_probs(probs, out_h, out_w)
        t_hi = float(np.percentile(p, 98.0))
        t_hi = max(0.5, min(0.99, t_hi))
        tight = (p > t_hi).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        tight = cv2.morphologyEx(tight, cv2.MORPH_OPEN, kernel)
        alt = _postprocess_mask(tight, morph=0, min_area=0, fill_holes=False, keep_largest=True)
        if 0 < np.count_nonzero(alt) < (out_h * out_w):
            return alt
    if np.count_nonzero(mask_np) == 0:
        p = _resize_probs(probs, out_h, out_w)
        prob_u8 = (p * 255.0).astype(np.uint8)
        _, otsu = cv2.threshold(prob_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        alt = _postprocess_mask(otsu, morph=0, min_area=0, fill_holes=False, keep_largest=keep_largest)
        if np.count_nonzero(alt) > 0:
            return alt
    if np.count_nonzero(mask_np) == 0:
        p = _resize_probs(probs, out_h, out_w)
        t = float(np.percentile(p, 90.0))
        t = max(0.2, min(0.9, t))
        dyn = (p > t).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        dyn = cv2.morphologyEx(dyn, cv2.MORPH_CLOSE, kernel)
        alt = _postprocess_mask(dyn, morph=0, min_area=0, fill_holes=False, keep_largest=keep_largest)
        if np.count_nonzero(alt) > 0:
            return alt
    return mask_np


def _read_rgb_cv(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Не удалось загрузить изображение: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def infer_segmentation_one(model: torch.nn.Module, image_path: Path, input_size: int, threshold: float, device: torch.device, *, debug: bool, use_clahe: bool, gamma: float, unsharp: float, tta: bool, adaptive: bool, morph: int, min_area: int, fill_holes: bool, keep_largest: bool) -> Tuple[np.ndarray, np.ndarray]:
    img_rgb = _read_rgb_cv(image_path)
    orig_h, orig_w = img_rgb.shape[:2]
    img_rgb = _apply_preprocessing(img_rgb, use_clahe, gamma, unsharp)
    x = _to_model_input(img_rgb, input_size, device)
    with torch.no_grad():
        if tta:
            preds = []
            logits = model(x)
            preds.append(logits)
            x_h = torch.flip(x, dims=[-1])
            preds.append(torch.flip(model(x_h), dims=[-1]))
            x_v = torch.flip(x, dims=[-2])
            preds.append(torch.flip(model(x_v), dims=[-2]))
            logits = torch.mean(torch.stack(preds, dim=0), dim=0)
        else:
            logits = model(x)
    mask = _logits_to_mask(logits, orig_h, orig_w, threshold, adaptive, morph, min_area, fill_holes, keep_largest, debug)
    return img_rgb, mask


def load_unetpp(checkpoint_path: Path, device: torch.device, *, arch: str, debug: bool) -> torch.nn.Module:
    # Используем локальную модель UNetPlusPlusOld как «old», иначе пытаемся импортировать новую, если есть
    try:
        from models.Unet_segmenter import UNetPlusPlusOld as LocalOld
        OldClass = LocalOld
    except Exception:
        from models.Unet_segmenter import UNetPlusPlusOld as OldClass
    if arch == 'old':
        model = OldClass(input_channels=3, output_channels=1, base_channels=64, depth=4,
                         attention_gates=False, deep_supervision=False, debug_mode=False)
    else:
        # На случай отсутствия «новой» архитектуры, используем old как fallback
        model = OldClass(input_channels=3, output_channels=1, base_channels=64, depth=4,
                         attention_gates=False, deep_supervision=False, debug_mode=False)
    model.to(device)
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        else:
            state = ckpt
    else:
        state = ckpt
    new_state = {}
    for k, v in state.items():
        if isinstance(k, str) and k.startswith('module.'):
            new_state[k[len('module.'):]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model
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


