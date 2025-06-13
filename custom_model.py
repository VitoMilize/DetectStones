import argparse
import os
import shutil

import cv2
import numpy as np
from ultralytics import YOLO

import logging
import os


def draw_segmented_objects(image, result):
    """
    Наносит на изображение маски с подписями вида: class_name (area px),
    и в верхнем левом углу отображает текущее количество объектов.
    """
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    h, w, _ = image.shape

    if result.masks is None or result.masks.data is None:
        cv2.putText(annotated, "Objects: 0", (10, 20), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        return annotated

    count = 0

    for mask, cls_id in zip(result.masks.data, result.boxes.cls):
        cls_name = result.names[int(cls_id)]
        binary_mask = mask.cpu().numpy().astype(np.uint8)
        area = np.sum(binary_mask)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)

        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            text = f"{cls_name} ({area} px)"
            cv2.putText(annotated, text, (x, y - 10), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            count += 1

    # Отображаем количество объектов в левом верхнем углу
    cv2.putText(annotated, f"Objects: {count}", (10, 20), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    return annotated


def setup_logger(log_file='train.log'):
    """
    Выдает логгер для работы
    """
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

    def train(self, data_path, epochs, batch):
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

    def evaluate(self, input_path, output_path, weights_path):
        """
        Обрабатывает изображения или видео из указанной папки и сохраняет результаты сегментации.

        :param input_path: Путь к файлу или папке с изображениями/видео.
        :param output_path: Папка для сохранения результатов.
        :param weights_path: Путь к весам, если нужны не дефолтные после обучения
        """
        self.logger.info(f"🚀 Начата оценка входных данных из {input_path}")

        if os.path.exists(output_path):
            self.logger.warning(f"⚠️ Удаляем старую папку результатов: {output_path}")
            shutil.rmtree(output_path)

        os.makedirs(output_path, exist_ok=True)

        self.model = YOLO(weights_path)

        input_files = []
        if os.path.isdir(input_path):
            for file in os.listdir(input_path):
                full_path = os.path.join(input_path, file)
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov', '.mkv')):
                    input_files.append(full_path)
        elif os.path.isfile(input_path):
            input_files = [input_path]
        else:
            self.logger.error("❌ Неверный путь к файлу или папке")
            return

        self.logger.info(f"🔍 Найдено файлов для обработки: {len(input_files)}")

        for path in input_files:
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.logger.info(f"🎥 Обработка видео: {path}")
                cap = cv2.VideoCapture(path)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out_path = os.path.join(output_path, os.path.basename(path))
                out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = self.model.predict(frame, imgsz=(1024, 1280))
                    annotated_frame = draw_segmented_objects(frame, results[0])
                    out.write(annotated_frame)

                    frame_count += 1
                    if frame_count % 10 == 0:
                        self.logger.debug(f"📸 Кадров обработано: {frame_count}")

                cap.release()
                out.release()
                self.logger.info(f"✅ Видео сохранено: {out_path}")
            else:
                image = cv2.imread(path)
                results = self.model.predict(image, imgsz=(1024, 1280))
                for r in results:
                    annotated = draw_segmented_objects(image, r)
                    filename = os.path.basename(path)
                    save_path = os.path.join(output_path, filename)
                    cv2.imwrite(save_path, annotated)
                    self.logger.info(f"✅ Сохранено: {save_path}")

        self.logger.info("🎉 Оценка завершена")

    def demo(self, input_path, weights_path):
        """
        Показывает в реальном времени сегментацию изображения, видео или всех файлов в папке.
        :param input_path: Путь к изображению или видео
        :param weights_path: Путь к весам модели
        """
        self.logger.info(f"👁️ Запуск демо-режима для: {input_path}")
        self.model = YOLO(weights_path)

        if not os.path.exists(input_path):
            self.logger.error("❌ Указанный путь не существует.")
            return

        def process_image(image):
            results = self.model.predict(image, imgsz=(1024, 1280))
            for r in results:
                annotated = draw_segmented_objects(image, r)
                cv2.imshow("Segmentation Demo", annotated)
                cv2.waitKey(0)
            cv2.destroyAllWindows()

        def process_video(video_path):
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                self.logger.error(f"❌ Не удалось открыть видео: {video_path}")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model.predict(frame, imgsz=(1024, 1280))
                for r in results:
                    annotated = draw_segmented_objects(frame, r)
                    cv2.imshow("Segmentation Demo", annotated)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        if os.path.isfile(input_path):
            if input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image = cv2.imread(input_path)
                process_image(image)
            elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                process_video(input_path)
            else:
                self.logger.error("❌ Поддерживаются только изображения и видео.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv8 Segmentation CLI')
    subparsers = parser.add_subparsers(dest='command', help='Команды: train / evaluate')

    # Подкоманда: train
    train_parser = subparsers.add_parser('train', help='Обучение модели')
    train_parser.add_argument('--data', type=str, default='dataset/data.yaml', help='Путь к data.yaml')
    train_parser.add_argument('--epochs', type=int, default=50, help='Количество эпох')
    train_parser.add_argument('--batch', type=int, default=16, help='Размер батча')

    # Подкоманда: evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Оценка на изображениях или видео')
    eval_parser.add_argument('--input', type=str, required=True, help='Путь к файлу или папке с изображениями/видео')
    eval_parser.add_argument('--output', type=str, default='./results', help='Папка для результатов')
    eval_parser.add_argument('--weights', type=str, default='./model/weights/best.pt', help='Путь к весам модели')

    demo_parser = subparsers.add_parser('demo', help='Демо-сегментация в реальном времени')
    demo_parser.add_argument('--input', type=str, required=True, help='Путь к файлу изображения или видео')
    demo_parser.add_argument('--weights', type=str, default='./model/weights/best.pt', help='Путь к весам модели')

    args = parser.parse_args()

    model = CustomModel()

    if args.command == 'train':
        model.train(
            data_path=args.data,
            epochs=args.epochs,
            batch=args.batch
        )
    elif args.command == 'evaluate':
        model.evaluate(
            input_path=args.input,
            output_path=args.output,
            weights_path=args.weights
        )
    elif args.command == 'demo':
        model.demo(
            input_path=args.input,
            weights_path=args.weights
        )
    else:
        parser.print_help()
