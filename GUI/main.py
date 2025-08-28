import sys
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QProgressBar, 
                             QTextEdit, QFrame, QFileDialog, QComboBox, QTreeView, QFileSystemModel, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDir
from PyQt5.QtGui import QFont, QPixmap, QPainter, QColor, QPalette

class CASMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CAS - Классификационно-сегментационный ИИ для онкологии кожи")
        self.setGeometry(100, 100, 1200, 800)
        
        # Устанавливаем стиль
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QFrame {
                background-color: #ffffff;
                border: 2px solid #d3d3d3;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #f8f9fa;
                color: #495057;
                border: 2px solid #d3d3d3;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
                border-color: #adb5bd;
            }
            QPushButton:pressed {
                background-color: #dee2e6;
                border-color: #6c757d;
            }
            QLabel {
                color: #333333;
                font-size: 14px;
                font-weight: bold;
            }
            QProgressBar {
                border: 2px solid #d3d3d3;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
            QTextEdit {
                border: 2px solid #d3d3d3;
                border-radius: 5px;
                background-color: #ffffff;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
            QComboBox {
                border: 2px solid #d3d3d3;
                border-radius: 5px;
                padding: 8px;
                background-color: white;
                min-height: 30px;
            }
        """)
        
        self.init_ui()
        # Состояние анализа
        self.analysis_running = False
        self.worker = None
        
    def init_ui(self):
        # Создаем центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Левая панель (контролы и настройки)
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Правая панель (изображения и логи)
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 3)
        
    def create_left_panel(self):
        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(15, 15, 15, 15)
        
        # Выбор модели (сплошная кнопка)
        self.model_button = QPushButton("Выберите модель")
        self.model_button.clicked.connect(self.select_model)
        left_layout.addWidget(self.model_button)
        
        # Выбор директории (сплошная кнопка)
        self.dir_button = QPushButton("Выберите директорию для анализа")
        self.dir_button.clicked.connect(self.select_directory)
        left_layout.addWidget(self.dir_button)
        
        # Выбор директории вывода (сплошная кнопка)
        self.output_dir_button = QPushButton("Выберите директорию вывода")
        self.output_dir_button.clicked.connect(self.select_output_directory)
        left_layout.addWidget(self.output_dir_button)
        
        # XAI Mode (сплошная кнопка-переключатель)
        self.xai_button = QPushButton("XAI Mode: Выключен")
        self.xai_button.setCheckable(True)
        self.xai_button.clicked.connect(self.toggle_xai)
        left_layout.addWidget(self.xai_button)
        
        # Удалено окно конфигурации системы
        
        # Навигация изображений
        nav_label = QLabel("Навигация по проекту:")
        left_layout.addWidget(nav_label)
        
        nav_frame = QFrame()
        nav_frame.setMinimumHeight(100)
        nav_layout = QVBoxLayout(nav_frame)

        # Дерево входной директории
        self.input_dir_title = QLabel("Входная директория")
        nav_layout.addWidget(self.input_dir_title)
        self.input_model = QFileSystemModel()
        self.input_model.setFilter(QDir.AllEntries | QDir.NoDotAndDotDot | QDir.AllDirs | QDir.Files)
        self.input_model.setNameFilters(["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"])
        self.input_model.setNameFilterDisables(False)
        self.input_tree = QTreeView()
        self.input_tree.setModel(self.input_model)
        self.input_tree.setHeaderHidden(True)
        self.input_tree.clicked.connect(self.on_input_tree_clicked)
        self.input_empty_label = QLabel("пусто")
        self.input_empty_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.input_empty_label)
        nav_layout.addWidget(self.input_tree)
        self.input_tree.setVisible(False)

        # Дерево директории вывода
        self.output_dir_title = QLabel("Директория вывода")
        nav_layout.addWidget(self.output_dir_title)
        self.output_model = QFileSystemModel()
        # Показываем только файлы (без директорий)
        self.output_model.setFilter(QDir.Files | QDir.NoDotAndDotDot)
        self.output_model.setNameFilters(["*_mask.png", "*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"])
        self.output_model.setNameFilterDisables(False)
        self.output_tree = QTreeView()
        self.output_tree.setModel(self.output_model)
        self.output_tree.setHeaderHidden(True)
        self.output_tree.clicked.connect(self.on_output_tree_clicked)
        self.output_empty_label = QLabel("пусто")
        self.output_empty_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.output_empty_label)
        nav_layout.addWidget(self.output_tree)
        self.output_tree.setVisible(False)

        # Дерево XAI результатов
        self.xai_dir_title = QLabel("XAI_results")
        nav_layout.addWidget(self.xai_dir_title)
        self.xai_model = QFileSystemModel()
        # Для XAI показываем и папки (по изображениям внутри них), и файлы
        self.xai_model.setFilter(QDir.AllDirs | QDir.Files | QDir.NoDotAndDotDot)
        self.xai_model.setNameFilters(["*.png", "*.jpg", "*.jpeg"])  # тепловые карты и т.п.
        self.xai_model.setNameFilterDisables(False)
        self.xai_tree = QTreeView()
        self.xai_tree.setModel(self.xai_model)
        self.xai_tree.setHeaderHidden(True)
        self.xai_tree.clicked.connect(self.on_xai_tree_clicked)
        self.xai_empty_label = QLabel("пусто")
        self.xai_empty_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.xai_empty_label)
        nav_layout.addWidget(self.xai_tree)
        self.xai_tree.setVisible(False)
        left_layout.addWidget(nav_frame)
        
        # Выбор устройства (показываем доступные)
        device_label = QLabel("Устройство (выбор):")
        left_layout.addWidget(device_label)
        
        self.device_combo = QComboBox()
        self.populate_available_devices()
        left_layout.addWidget(self.device_combo)
        
        # Кнопка запуска
        self.run_button = QPushButton("Запустить анализ")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: 2px solid #495057;
                font-size: 14px;
                min-height: 50px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a6268;
                border-color: #343a40;
            }
            QPushButton:pressed {
                background-color: #495057;
                border-color: #212529;
            }
        """)
        self.run_button.clicked.connect(self.run_analysis)
        left_layout.addWidget(self.run_button)
        
        # Добавляем растягивающийся элемент в конец
        left_layout.addStretch()
        
        return left_frame
        
    def create_right_panel(self):
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(15, 15, 15, 15)
        
        # Верхняя панель с изображениями (занимает 60% пространства)
        images_panel = self.create_images_panel()
        right_layout.addWidget(images_panel, 6)
        
        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(45)  # Пример значения
        self.progress_bar.setMinimumHeight(30)
        self.progress_bar.setMaximumHeight(30)
        right_layout.addWidget(self.progress_bar)
        
        # Логи (занимают оставшееся место - 40% пространства)
        self.logs_text = QTextEdit()
        self.logs_text.setPlainText("")  # Пустое место
        right_layout.addWidget(self.logs_text, 4)
        
        return right_frame
        
    def create_images_panel(self):
        images_frame = QFrame()
        images_layout = QHBoxLayout(images_frame)
        images_layout.setSpacing(20)
        
        # Левое изображение (исходное)
        left_image_frame = QFrame()
        left_image_layout = QVBoxLayout(left_image_frame)
        
        self.left_image_label = QLabel("исходное изображение")
        self.left_image_label.setAlignment(Qt.AlignCenter)
        self.left_image_label.setStyleSheet("""
            font-size: 16px;
            color: #7f8c8d;
            background-color: #ecf0f1;
            border: 2px dashed #bdc3c7;
            border-radius: 5px;
            padding: 20px;
        """)
        # Ограничиваем рост и позволяем свободное масштабирование содержимого
        self.left_image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.left_image_label.setMinimumSize(200, 200)
        self.left_image_label.setMaximumHeight(420)
        left_image_layout.addWidget(self.left_image_label)
        
        images_layout.addWidget(left_image_frame)
        
        # Правое изображение (предсказанная маска)
        right_image_frame = QFrame()
        right_image_layout = QVBoxLayout(right_image_frame)
        
        self.right_image_label = QLabel("Предсказан-ная маска")
        self.right_image_label.setAlignment(Qt.AlignCenter)
        self.right_image_label.setStyleSheet("""
            font-size: 16px;
            color: #7f8c8d;
            background-color: #ecf0f1;
            border: 2px dashed #bdc3c7;
            border-radius: 5px;
            padding: 20px;
        """)
        # Ограничиваем рост и позволяем свободное масштабирование содержимого
        self.right_image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.right_image_label.setMinimumSize(200, 200)
        self.right_image_label.setMaximumHeight(420)
        right_image_layout.addWidget(self.right_image_label, 1)
        
        # Предсказанный класс (маленький)
        self.class_label = QLabel("Предсказанный класс: Не определен")
        self.class_label.setAlignment(Qt.AlignCenter)
        self.class_label.setStyleSheet("""
            font-size: 12px;
            color: #2c3e50;
            background-color: #e8f5e8;
            border: 1px solid #27ae60;
            border-radius: 3px;
            padding: 5px;
            margin-top: 10px;
            max-height: 35px;
            min-height: 35px;
        """)
        right_image_layout.addWidget(self.class_label, 0)
        
        images_layout.addWidget(right_image_frame)
        
        return images_frame

    # ===== Навигация: обработчики кликов =====
    def on_input_tree_clicked(self, index):
        try:
            path = self.input_model.filePath(index)
            from pathlib import Path
            p = Path(path)
            if p.is_file():
                self._show_for_input_image(p)
        except Exception as e:
            self.logs_text.append(f"[Навигация вход] {e}")

    def on_output_tree_clicked(self, index):
        try:
            path = self.output_model.filePath(index)
            from pathlib import Path
            p = Path(path)
            if p.is_file():
                if p.name.endswith('_mask.png'):
                    self._show_for_mask(p)
                else:
                    # если выбрали изображение в выходе, пытаемся найти его маску
                    stem = p.stem
                    # дерево теперь указывает на директорию вывода/masks, поэтому ищем относительно корня вывода
                    out_dir = Path(getattr(self, 'output_directory', ''))
                    mask = (out_dir / 'masks') / f"{stem}_mask.png"
                    if mask.exists():
                        self._show_for_mask(mask)
                    else:
                        self._show_for_input_image(p)
        except Exception as e:
            self.logs_text.append(f"[Навигация вывод] {e}")

    def on_xai_tree_clicked(self, index):
        try:
            path = self.xai_model.filePath(index)
            from pathlib import Path
            p = Path(path)
            if p.is_file():
                # Показываем слева исходник: ищем файл по имени папки (stem) в входной директории
                folder = p.parent
                base_stem = folder.name
                from pathlib import Path as _P
                inp_dir = getattr(self, 'input_directory', None)
                if inp_dir:
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        cand = _P(inp_dir) / f"{base_stem}{ext}"
                        if cand.exists():
                            self._set_label_pixmap(self.left_image_label, cand)
                            break
                # Правое окно — выбранное XAI изображение
                self._set_label_pixmap(self.right_image_label, p)
                # Обновим класс по CSV
                try:
                    stem_guess = folder.name
                    self._load_predictions_map()
                    cls_name = getattr(self, '_pred_map', {}).get(stem_guess, 'Не определен')
                    self.class_label.setText(f"Предсказанный класс: {cls_name}")
                except Exception:
                    pass
        except Exception as e:
            self.logs_text.append(f"[Навигация XAI] {e}")

    # ===== Служебные методы =====
    def _update_tree_roots(self):
        from pathlib import Path
        if hasattr(self, 'input_directory') and self.input_directory:
            root_idx = self.input_model.setRootPath(self.input_directory)
            self.input_tree.setRootIndex(root_idx)
            # Скрываем колонки кроме имени
            for c in range(1, 4):
                self.input_tree.setColumnHidden(c, True)
            self.input_tree.setVisible(True)
            self.input_empty_label.setVisible(False)
        else:
            self.input_tree.setVisible(False)
            self.input_empty_label.setVisible(True)
        if hasattr(self, 'output_directory') and self.output_directory:
            # Корень дерева масок: директорияВывода/masks
            from pathlib import Path as _P
            out = _P(self.output_directory)
            (out / 'masks').mkdir(parents=True, exist_ok=True)
            masks_path = str(out / 'masks')
            root_idx2 = self.output_model.setRootPath(masks_path)
            self.output_tree.setRootIndex(root_idx2)
            for c in range(1, 4):
                self.output_tree.setColumnHidden(c, True)
            self.output_tree.setVisible(True)
            self.output_empty_label.setVisible(False)
            # Корень XAI: создаём подпапку XAI_results при необходимости
            try:
                (out / 'XAI_results').mkdir(parents=True, exist_ok=True)
                # Привязываем XAI к директории вывода/XAI_results
                xai_path = str(out / 'XAI_results')
                xai_root_idx = self.xai_model.setRootPath(xai_path)
                self.xai_tree.setRootIndex(xai_root_idx)
                for c in range(1, 4):
                    self.xai_tree.setColumnHidden(c, True)
                self.xai_tree.setVisible(True)
                self.xai_empty_label.setVisible(False)
            except Exception:
                self.xai_tree.setVisible(False)
                self.xai_empty_label.setVisible(True)
        else:
            self.output_tree.setVisible(False)
            self.output_empty_label.setVisible(True)
            self.xai_tree.setVisible(False)
            self.xai_empty_label.setVisible(True)

    def _load_predictions_map(self):
        # Кэшируем predictions.csv
        from pathlib import Path
        import csv
        output_dir = getattr(self, 'output_directory', None)
        if not output_dir:
            self._pred_map = {}
            return
        csv_path = Path(output_dir) / 'predictions.csv'
        if not csv_path.exists():
            self._pred_map = {}
            return
        try:
            mapping = {}
            with csv_path.open('r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and len(header) >= 2:
                    classes = header[1:]
                else:
                    classes = []
                for row in reader:
                    if not row:
                        continue
                    stem = row[0]
                    vals = [float(x) for x in row[1:]] if len(row) > 1 else []
                    if vals and classes:
                        idx = int(max(range(len(vals)), key=lambda i: vals[i]))
                        mapping[stem] = classes[idx]
            self._pred_map = mapping
        except Exception:
            self._pred_map = {}

    def _find_input_image_by_stem(self, stem: str):
        from pathlib import Path
        if not getattr(self, 'input_directory', None):
            return None
        base = Path(self.input_directory)
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            p = base / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    def _find_mask_for_stem(self, stem: str):
        """Ищем маску для изображения по stem в директории вывода.

        Поддерживаем несколько мест и имён: masks/<stem>_mask.png, <stem>_mask.png,
        а также рекурсивный поиск по подпапкам, если не найдено напрямую.
        """
        from pathlib import Path
        out_dir = getattr(self, 'output_directory', None)
        if not out_dir:
            return None
        base = Path(out_dir)
        # 1) masks/<stem>_mask.png
        cand = base / 'masks' / f"{stem}_mask.png"
        if cand.exists():
            return cand
        # 2) <stem>_mask.png в корне вывода
        cand2 = base / f"{stem}_mask.png"
        if cand2.exists():
            return cand2
        # 3) Рекурсивный поиск по поддиректориям
        try:
            for p in base.rglob(f"{stem}_mask.*"):
                if p.is_file():
                    return p
        except Exception:
            pass
        return None

    def _set_label_pixmap(self, label: QLabel, image_path):
        from PyQt5.QtGui import QPixmap
        pix = QPixmap(str(image_path))
        if pix.isNull():
            label.setText("Не удалось загрузить")
            return
        # Масштабируем в границах текущего размера лейбла, не меняя размер окна
        target_w = max(1, label.width())
        target_h = max(1, label.height())
        label.setPixmap(pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _show_for_input_image(self, image_path):
        # левое: изображение, правое: маска, класс — из CSV
        from pathlib import Path
        p = Path(image_path)
        self._set_label_pixmap(self.left_image_label, p)
        # маска
        out_dir = getattr(self, 'output_directory', None)
        if out_dir:
            mask_p = self._find_mask_for_stem(p.stem)
            if mask_p:
                self._set_label_pixmap(self.right_image_label, mask_p)
            else:
                self.right_image_label.setText("Маска не найдена")
        else:
            self.right_image_label.setText("Не выбрана директория вывода")
        # класс
        self._load_predictions_map()
        cls_name = getattr(self, '_pred_map', {}).get(p.stem, 'Не определен')
        self.class_label.setText(f"Предсказанный класс: {cls_name}")

    def _show_for_mask(self, mask_path):
        from pathlib import Path
        mp = Path(mask_path)
        self._set_label_pixmap(self.right_image_label, mp)
        # изображение
        img_p = self._find_input_image_by_stem(mp.stem.replace('_mask', ''))
        if img_p:
            self._set_label_pixmap(self.left_image_label, img_p)
            stem = img_p.stem
        else:
            self.left_image_label.setText("Исходное изображение не найдено")
            stem = mp.stem.replace('_mask', '')
        # класс
        self._load_predictions_map()
        cls_name = getattr(self, '_pred_map', {}).get(stem, 'Не определен')
        self.class_label.setText(f"Предсказанный класс: {cls_name}")
        
    def populate_available_devices(self):
        """Определяем доступные устройства"""
        devices = ["CPU"]
        
        # Проверяем доступность CUDA
        if torch.cuda.is_available():
            devices.append("CUDA")
            # Добавляем информацию о GPU
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                devices.append(f"GPU {i}: {gpu_name}")
        
        self.device_combo.addItems(devices)
        
    def select_model(self):
        """Выбор директории с чекпойнтами моделей (сегментация и классификация)"""
        directory = QFileDialog.getExistingDirectory(self, "Выберите директорию с чекпойнтами моделей")
        if not directory:
            return
        self.model_directory = directory
        self.logs_text.append(f"Выбрана директория моделей: {directory}")

        # Автодетект файлов чекпойнтов: ищем .pth похожие на segmentation / classification
        from pathlib import Path
        model_dir = Path(directory)
        pth_files = list(model_dir.glob('*.pth'))
        seg_ckpt = None
        cls_ckpt = None
        # простые эвристики по имени
        for f in pth_files:
            name = f.name.lower()
            if seg_ckpt is None and any(k in name for k in ["seg", "segment", "unet"]):
                seg_ckpt = f
            if cls_ckpt is None and any(k in name for k in ["cls", "class", "resnet"]):
                cls_ckpt = f
        # если не нашли по эвристике, просто назначим первые два по размеру/имени
        if seg_ckpt is None and pth_files:
            seg_ckpt = sorted(pth_files, key=lambda x: x.stat().st_size, reverse=True)[0]
        if cls_ckpt is None and len(pth_files) > 1:
            # следующий по размеру
            sorted_files = sorted(pth_files, key=lambda x: x.stat().st_size, reverse=True)
            # если первый ушёл под seg_ckpt
            candidates = [p for p in sorted_files if p != seg_ckpt]
            if candidates:
                cls_ckpt = candidates[0]

        if seg_ckpt:
            self.segmentation_checkpoint = str(seg_ckpt)
            self.logs_text.append(f"Сегментация чекпойнт: {seg_ckpt}")
        else:
            self.logs_text.append("[Предупреждение] Не найден чекпойнт сегментации (.pth)")

        if cls_ckpt:
            self.classification_checkpoint = str(cls_ckpt)
            self.logs_text.append(f"Классификация чекпойнт: {cls_ckpt}")
        else:
            self.logs_text.append("[Предупреждение] Не найден чекпойнт классификации (.pth)")
        
    def select_directory(self):
        """Заглушка для выбора директории"""
        directory = QFileDialog.getExistingDirectory(self, "Выберите директорию для анализа")
        if directory:
            self.logs_text.append(f"Выбрана директория: {directory}")
            self.input_directory = directory
            self._update_tree_roots()
            
    def select_output_directory(self):
        """Выбор директории вывода"""
        directory = QFileDialog.getExistingDirectory(self, "Выберите директорию вывода")
        if directory:
            self.logs_text.append(f"Выбрана директория вывода: {directory}")
            self.output_directory = directory
            self._update_tree_roots()
            
    def toggle_xai(self):
        """Переключение XAI режима"""
        if self.xai_button.isChecked():
            self.xai_button.setText("XAI Mode: Включен")
            self.logs_text.append("XAI режим включен")
        else:
            self.xai_button.setText("XAI Mode: Выключен")
            self.logs_text.append("XAI режим выключен")
            
    def run_analysis(self):
        """Запуск пайплайна сегментации и классификации"""
        from pathlib import Path
        # Гарантируем доступность корня проекта для импорта core/
        import sys as _sys
        _root = Path(__file__).resolve().parent.parent
        if str(_root) not in _sys.path:
            _sys.path.append(str(_root))
        from core.pipeline import CASPipeline, PipelineConfig

        # Если анализ уже запущен — останавливаем
        if getattr(self, 'analysis_running', False) and getattr(self, 'worker', None) and self.worker.isRunning():
            try:
                self.logs_text.append("Остановка анализа...")
                if hasattr(self.worker, 'stop'):
                    self.worker.stop()
                else:
                    self.worker.requestInterruption()
                # Кнопку сразу вернем в исходное состояние
                self.run_button.setText("Запустить анализ")
                self.analysis_running = False
            except Exception as e:
                self.logs_text.append(f"[Ошибка остановки] {e}")
            return

        # Проверка входа/выхода
        input_dir = getattr(self, 'input_directory', None)
        output_dir = getattr(self, 'output_directory', None)
        if not input_dir:
            self.logs_text.append("[Ошибка] Не выбрана директория для анализа")
            return
        if not output_dir:
            self.logs_text.append("[Ошибка] Не выбрана директория вывода")
            return

        # Параметры моделей: берем из выбранной директории, если заданы
        from pathlib import Path
        seg_ckpt = Path(getattr(self, 'segmentation_checkpoint', ''))
        cls_ckpt = Path(getattr(self, 'classification_checkpoint', ''))
        if not seg_ckpt.exists() or not seg_ckpt.is_file():
            self.logs_text.append("[Ошибка] Не выбран или не найден чекпойнт сегментации. Используйте 'Выберите модель'.")
            return
        if not cls_ckpt.exists() or not cls_ckpt.is_file():
            self.logs_text.append("[Ошибка] Не выбран или не найден чекпойнт классификации. Используйте 'Выберите модель'.")
            return

        # Определяем устройство по выбору пользователя
        sel = self.device_combo.currentText() if hasattr(self, 'device_combo') else 'CPU'
        device_opt = 'cuda' if ('CUDA' in sel or 'GPU' in sel) else 'cpu'

        # Пробрасываем XAI режим в пайплайн
        xai_on = bool(self.xai_button.isChecked()) if hasattr(self, 'xai_button') else False
        cfg = PipelineConfig(
            seg_checkpoint=seg_ckpt,
            cls_checkpoint=cls_ckpt,
            output_dir=Path(output_dir),
            device=device_opt,
            xai_enabled=xai_on
        )

        # Запуск в отдельном потоке с прогрессом
        self.logs_text.append("Запуск анализа...")
        self.progress_bar.setValue(0)

        # Определяем воркер
        class AnalysisWorker(QThread):
            progress = pyqtSignal(int)
            log = pyqtSignal(str)
            finished_ok = pyqtSignal(int)
            failed = pyqtSignal(str)

            def __init__(self, cfg: PipelineConfig, input_dir: Path):
                super().__init__()
                self.cfg = cfg
                self.input_dir = input_dir
                self._stop = False

            def stop(self):
                self._stop = True
                self.requestInterruption()

            def run(self):
                try:
                    # Lazy import внутри потока
                    from core.pipeline import CASPipeline
                    pipeline = CASPipeline(self.cfg)

                    # Список файлов для оценки прогресса
                    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
                    files = []
                    for e in exts:
                        files.extend(Path(self.input_dir).rglob(e))
                    files = sorted(files)
                    total = len(files)
                    if total == 0:
                        self.failed.emit("В директории нет изображений")
                        return

                    processed = 0
                    for idx, p in enumerate(files, start=1):
                        if self.isInterruptionRequested() or self._stop:
                            break
                        try:
                            pipeline.process_image(p)
                        except Exception as e:
                            self.log.emit(f"Ошибка обработки {p.name}: {e}")
                        prog = int(idx * 100 / total)
                        self.progress.emit(prog)
                        processed = idx
                    # Если прервано — сообщаем, сколько успели
                    if self.isInterruptionRequested() or self._stop:
                        self.finished_ok.emit(processed)
                        return
                    self.finished_ok.emit(total)
                except Exception as e:
                    self.failed.emit(str(e))

        self.worker = AnalysisWorker(cfg, Path(input_dir))
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.logs_text.append)
        def _ok(n):
            self.logs_text.append(f"Обработано изображений: {n}")
            self.logs_text.append(f"Маски и CSV сохранены в: {cfg.output_dir}")
            self.progress_bar.setValue(100)
            # Сброс состояния и кнопки
            self.analysis_running = False
            self.run_button.setText("Запустить анализ")
        self.worker.finished_ok.connect(_ok)
        def _fail(m):
            self.logs_text.append(f"[Ошибка анализа] {m}")
            self.analysis_running = False
            self.run_button.setText("Запустить анализ")
        self.worker.failed.connect(_fail)
        self.worker.start()
        # Обновляем кнопку и флаг
        self.analysis_running = True
        self.run_button.setText("Остановить анализ")

def main():
    app = QApplication(sys.argv)
    
    # Устанавливаем глобальные стили для приложения
    app.setStyle('Fusion')
    
    window = CASMainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback, os
        log_path = os.path.join(os.getcwd(), 'gui_error.log')
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(''.join(traceback.format_exc()))
        except Exception:
            pass
        try:
            from PyQt5.QtWidgets import QMessageBox, QApplication
            app = QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(None, "Ошибка запуска", f"Приложение завершилось с ошибкой.\nЛог: {log_path}")
        except Exception:
            pass
        print("Fatal error. See gui_error.log", file=sys.stderr)
        raise