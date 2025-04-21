import argparse
import os
import shutil

from ultralytics import YOLO

import logging
import os


def setup_logger(log_file='train.log'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger('yolo_trainer')
    logger.setLevel(logging.DEBUG)

    # Формат логов
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")

    # Файл
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


class CustomModel:
    def __init__(self):
        self.model = YOLO("models/yolov8n-seg.pt")
        self.logger = setup_logger('./data/log_file.log')

    def train(self, data_path='./dataset/data.yaml', epochs=50, batch=16):
        """
        Обучает модель YOLO на задаче сегментации и сохраняет лучшие веса.
        """

        runs_path = './runs'
        source_path = './runs/segment/train'
        target_path = './model'

        if os.path.exists(target_path):
            print(f"Удаляем предыдущую папку: {target_path}")
            shutil.rmtree(target_path)
            self.logger.info(f"Папка {target_path} удалена")

        if os.path.exists(runs_path):
            self.logger.warning(f"⚠️ Удаляем предыдущую папку: {runs_path}")
            shutil.rmtree(runs_path)
            self.logger.info(f"Папка {runs_path} удалена")

        self.logger.warning("Обучение началось")

        self.model.train(
            task='segment',
            data=data_path,
            epochs=epochs,
            batch=batch,
            imgsz=(1024, 1280),
            save=True
        )

        self.logger.info("Обучение завершено")

        os.makedirs('./model', exist_ok=True)

        for item in os.listdir(source_path):
            src = os.path.join(source_path, item)
            dst = os.path.join(target_path, item)
            shutil.move(src, dst)

        shutil.rmtree('./runs')

        self.logger.info(f" Результаты обучения сохранены в папку: {target_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv8 segmentation model.')

    parser.add_argument('--data', type=str, default='dataset/data.yaml', help='Путь к data.yaml')
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох')
    parser.add_argument('--batch', type=int, default=16, help='Размер батча')

    args = parser.parse_args()

    model = CustomModel()
    model.train(
        data_path=args.data,
        epochs=args.epochs,
        batch=args.batch
    )
