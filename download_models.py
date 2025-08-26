#!/usr/bin/env python3
"""
Скрипт для скачивания чекпойнтов моделей CAS с Google Drive.

Скачивает:
- segmentation.pth (сегментация)
- classification.pth (классификация)

Файлы сохраняются в папку checkpoints/
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm


def download_file_from_drive(file_id: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Скачивает файл с Google Drive по ID."""
    try:
        # Google Drive direct download URL
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        
        print(f"Скачиваю {output_path.name}...")
        
        # Создаём директорию, если её нет
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Скачиваем с прогресс-баром
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Получаем размер файла
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✓ {output_path.name} успешно скачан")
        return True
        
    except Exception as e:
        print(f"✗ Ошибка скачивания {output_path.name}: {e}")
        return False


def main():
    print("=" * 60)
    print("CAS Models Downloader")
    print("Скачивание чекпойнтов моделей")
    print("=" * 60)
    
    # ID файлов с Google Drive
    models = {
        "segmentation.pth": "1rF6OdUCzO4-tTFx9riv8Ct2o0ooZbNct",
        "classification.pth": "1rF6OdUCzO4-tTFx9riv8Ct2o0ooZbNct"  # Пока тот же ID
    }
    
    # Папка для сохранения
    checkpoints_dir = Path("checkpoints")
    
    print(f"Чекпойнты будут сохранены в: {checkpoints_dir.absolute()}")
    print()
    
    # Проверяем зависимости
    try:
        import requests
        import tqdm
    except ImportError:
        print("Устанавливаю необходимые зависимости...")
        os.system(f"{sys.executable} -m pip install requests tqdm")
        print()
    
    # Скачиваем модели
    success_count = 0
    for filename, file_id in models.items():
        output_path = checkpoints_dir / filename
        
        if output_path.exists():
            print(f"⚠ {filename} уже существует, пропускаю")
            success_count += 1
            continue
        
        if download_file_from_drive(file_id, output_path):
            success_count += 1
    
    print()
    print("=" * 60)
    if success_count == len(models):
        print("✓ Все модели успешно скачаны!")
        print(f"Файлы находятся в: {checkpoints_dir.absolute()}")
    else:
        print(f"⚠ Скачано {success_count}/{len(models)} моделей")
        print("Проверьте ошибки выше")
    print("=" * 60)


if __name__ == "__main__":
    main()
