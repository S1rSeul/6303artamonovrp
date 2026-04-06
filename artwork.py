import csv
import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

import cv2

import numpy as np
from numpy.typing import NDArray

import requests


ImageU8 = NDArray[np.uint8]
ImageF32 = NDArray[np.float32]


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def timeit(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logging.info(f"[TIME] {func.__name__} выполнена за {end - start:.6f} секунд")
        return result
    return wrapper


def get_painting_id(csv_path: str) -> str:
    if not hasattr(get_painting_id, 'paintings'):
        paintings = []
        with open(csv_path, mode='r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if (row.get('Classification') == 'Paintings'
                        and row.get('Is Public Domain') == 'True'):
                    paintings.append(row.get("Object ID"))
        get_painting_id.paintings = paintings

    return random.choice(get_painting_id.paintings)


def fetch_object_metadata(object_id: str) -> dict:
    url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def download_image(image_url: str, save_path: str) -> None:
    response = requests.get(image_url)
    response.raise_for_status()

    with open(save_path, 'wb') as f:
        f.write(response.content)


def save_metadata(data: dict, save_path: str) -> None:
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class Artwork(ABC):
    __slots__ = ('_image', '_metadata')

    @property
    def image(self) -> ImageU8:
        return self._image.copy()

    @property
    def metadata(self) -> dict:
        return self._metadata.copy()

    def __init__(self, image: ImageU8, metadata: dict):
        self._image = image
        self._metadata = metadata

    def __str__(self) -> str:
        title = self.metadata.get('title', 'Неизвестен')
        artist = self.metadata.get('artistDisplayName', 'Неизвестен')
        return f"{self.__class__.__name__}: '{title}' by {artist}"

    @timeit
    def __add__(self, other: 'Artwork') -> 'Artwork':
        if not isinstance(other, Artwork):
            raise TypeError("Можно складывать только объекты Artwork")

        if self.metadata != other.metadata:
            raise ValueError("Можно складывать только изображения с одинаковыми метаданными")

        image1 = self.image
        image2 = other.image

        if image1.ndim != image2.ndim:
            if image1.ndim == 3 and image2.ndim == 2:
                image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR).astype(np.uint8)
            elif image1.ndim == 2 and image2.ndim == 3:
                image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR).astype(np.uint8)
            else:
                raise ValueError("Несовместимые размерности изображений")

        new_image = cv2.add(image1, image2)
        if isinstance(self, ColorArtwork) or isinstance(other, ColorArtwork):
            result_class = ColorArtwork
        else:
            result_class = GrayscaleArtwork
        return result_class(new_image, self.metadata)

    def _convolve_array(self, kernel: ImageF32, astype: str = 'int', method: str = 'manual') -> ImageU8 | ImageF32:
        if method == 'manual':
            k_h, k_w = kernel.shape
            pad_h, pad_w = k_h // 2, k_w // 2

            padded_width = [(pad_h, pad_h), (pad_w, pad_w)]
            if self.image.ndim == 3:
                padded_width.append((0, 0))

            padded = np.pad(self.image, padded_width, mode='reflect')

            windows = np.lib.stride_tricks.sliding_window_view(padded, (k_h, k_w), axis=(0, 1))
            result = np.tensordot(windows, kernel, axes=((-2, -1), (0, 1)))

            if astype == 'int':
                return np.clip(result, 0, 255).astype(np.uint8)
            else:
                return result.astype(np.float32)
        elif method == 'opencv':
            return cv2.filter2D(self.image, ddepth=-1, kernel=kernel)
        else:
            raise ValueError("method должен быть 'manual' или 'opencv'")

    @timeit
    def convolve(self, kernel: ImageF32, astype: str = 'int', method: str = 'manual') -> 'Artwork':
        result = self._convolve_array(kernel, astype, method)
        return self.__class__(result, self.metadata)

    @timeit
    def gaussian(self, ksize: int, sigma: float, method: str = 'manual') -> 'Artwork':
        if method == 'manual':
            k = ksize // 2
            x, y = np.mgrid[-k:k + 1, -k:k + 1]
            kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            kernel /= kernel.sum()
            blurred = self._convolve_array(kernel)
            return self.__class__(blurred, self.metadata)
        elif method == 'opencv':
            blurred = cv2.GaussianBlur(self.image, (ksize, ksize), sigma)
            return self.__class__(blurred, self.metadata)
        else:
            raise ValueError("method должен быть 'manual' или 'opencv'")

    @timeit
    def sobel(self, method: str = 'manual') -> 'Artwork':
        if method == 'manual':
            sobel_x = np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ], dtype=np.float32)

            sobel_y = np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1],
            ], dtype=np.float32)

            gx = self._convolve_array(sobel_x, 'float')
            gy = self._convolve_array(sobel_y, 'float')

            magnitude = np.sqrt(gx ** 2 + gy ** 2).astype(np.uint8)
            return self.__class__(magnitude, self.metadata)
        elif method == 'opencv':
            gx = cv2.Sobel(self.image, ddepth=cv2.CV_32F, dx=1, dy=0)
            gy = cv2.Sobel(self.image, ddepth=cv2.CV_32F, dx=0, dy=1)

            magnitude = cv2.magnitude(gx, gy).astype(np.uint8)
            return self.__class__(magnitude, self.metadata)
        else:
            raise ValueError("method должен быть 'manual' или 'opencv'")

    @timeit
    def gamma_correction(self, gamma: float, method: str = 'manual') -> 'Artwork':
        if method == 'manual':
            image = self.image.astype(np.float32) / 255.0
            corrected = np.power(image, 1 / gamma)
            result = (corrected * 255).astype(np.uint8)
        elif method == 'opencv':
            image = self.image.astype(np.float32) / 255.0
            corrected = cv2.pow(image, 1 / gamma)
            result = (corrected * 255).astype(np.uint8)
        else:
            raise ValueError("method должен быть 'manual' или 'opencv'")
        return self.__class__(result, self.metadata)

    @abstractmethod
    def grayscale(self, method: str = 'manual') -> 'Artwork':
        pass

    @abstractmethod
    def equalize_hist(self, method: str = 'manual') -> 'Artwork':
        pass


class ColorArtwork(Artwork):
    __slots__ = ()

    def __init__(self, image: ImageU8, metadata: dict):
        if image.ndim != 3:
            raise ValueError("ColorArtwork ожидает 3-канальное изображение")
        super().__init__(image, metadata)

    @timeit
    def grayscale(self, method: str = 'manual') -> 'GrayscaleArtwork':
        if method == 'manual':
            weights = np.array((0.114, 0.587, 0.299), dtype=np.float32)
            gray = np.clip(self.image @ weights, 0, 255).astype(np.uint8)
        elif method == 'opencv':
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("method должен быть 'manual' или 'opencv'")
        return GrayscaleArtwork(gray, self.metadata)

    @timeit
    def equalize_hist(self, method: str = 'manual') -> 'ColorArtwork':
        if method == 'manual':
            lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_eq = self._equalize_hist_impl(l)
        elif method == 'opencv':
            lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_eq = cv2.equalizeHist(l)
        else:
            raise ValueError("method должен быть 'manual' или 'opencv'")
        lab_eq = cv2.merge([l_eq, a, b])
        result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        return ColorArtwork(result, self.metadata)

    @staticmethod
    def _equalize_hist_impl(image: ImageU8) -> ImageU8:
        hist = np.histogram(image.flatten(), 256, (0, 256))[0]
        cdf = hist.cumsum()
        cdf_norm = cdf * 255 / cdf[-1]
        lut = np.round(cdf_norm).astype(np.uint8)
        return lut[image]


class GrayscaleArtwork(Artwork):
    __slots__ = ()

    def __init__(self, image: ImageU8, metadata: dict):
        if image.ndim != 2:
            raise ValueError("GrayscaleArtwork ожидает 2-мерное изображение")
        super().__init__(image, metadata)

    @timeit
    def grayscale(self, method: str = 'manual') -> 'GrayscaleArtwork':
        return GrayscaleArtwork(self.image, self.metadata)

    @timeit
    def equalize_hist(self, method: str = 'manual') -> 'GrayscaleArtwork':
        if method == 'manual':
            hist = np.histogram(self.image.flatten(), 256, (0, 256))[0]
            cdf = hist.cumsum()
            cdf_norm = cdf * 255 / cdf[-1]
            lut = np.round(cdf_norm).astype(np.uint8)
            result = lut[self.image]
        elif method == 'opencv':
            result = cv2.equalizeHist(self.image)
        else:
            raise ValueError("method должен быть 'manual' или 'opencv'")
        return GrayscaleArtwork(result, self.metadata)


class ImageProcessor:
    __slots__ = ('_csv_path', '_output_dir')

    def __init__(self, csv_path: str = 'MetObjects.csv', output_dir: str = 'paintings'):
        self._csv_path = csv_path
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)

    @timeit
    def download_random_painting(self) -> Artwork:
        object_id = get_painting_id(self._csv_path)
        logging.info(f"Выбрана картина ID {object_id}")
        logging.info(f"Загрузка метаданных для объекта {object_id}")
        metadata = fetch_object_metadata(object_id)

        primary_image = metadata.get('primaryImage')
        if not primary_image:
            raise ValueError(f"У объекта {object_id} отсутствует primaryImage")

        img_path = os.path.join(self._output_dir, 'image.jpg')
        logging.info(f"Скачивание изображения: {primary_image}")
        download_image(primary_image, img_path)
        logging.info(f"Изображение сохранено в {img_path}")

        image = cv2.imread(img_path)

        json_path = os.path.join(self._output_dir, 'image.json')
        save_metadata(metadata, json_path)
        logging.info(f"Метаданные сохранены в {json_path}")

        if image.ndim == 3:
            artwork = ColorArtwork(image, metadata)
        elif image.ndim == 2:
            artwork = GrayscaleArtwork(image, metadata)
        else:
            raise ValueError(f"Неподдерживаемая размерность изображения: {image.shape}")

        logging.info(f"Создан объект: {artwork}")
        return artwork

    @timeit
    def process_artwork(self, artwork: Artwork, prefix: str = '') -> None:
        logging.info(f"Начало обработки изображения с префиксом '{prefix}'...")

        orig_path = os.path.join(self._output_dir, f'image_{prefix}_original.jpg')
        cv2.imwrite(orig_path, artwork.image)
        logging.info(f"Оригинал изображения сохранен в {orig_path}")

        sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ], dtype=np.float32)
        ksize = 3
        sigma = 1.0
        gamma = 0.5

        operations_manual = [
            (artwork.grayscale, 'grayscale_manual', {'method': 'manual'}),
            (artwork.convolve, 'convolve_manual', {'kernel': sharpen_kernel, 'method': 'manual'}),
            (artwork.gaussian, f'gaussian_manual_ks{ksize}_s{sigma}', {'ksize': ksize, 'sigma': sigma, 'method': 'manual'}),
            (artwork.sobel, 'sobel_mag_manual', {'method': 'manual'}),
            (artwork.gamma_correction, f'gamma_manual_g{gamma}', {'gamma': gamma, 'method': 'manual'}),
            (artwork.equalize_hist, 'eq_hist_manual', {'method': 'manual'}),
        ]

        operations_opencv = [
            (artwork.grayscale, 'grayscale_opencv', {'method': 'opencv'}),
            (artwork.convolve, 'convolve_opencv', {'kernel': sharpen_kernel, 'method': 'opencv'}),
            (artwork.gaussian, f'gaussian_opencv_ks{ksize}_s{sigma}', {'ksize': ksize, 'sigma': sigma, 'method': 'opencv'}),
            (artwork.sobel, 'sobel_mag_opencv', {'method': 'opencv'}),
            (artwork.gamma_correction, f'gamma_opencv_g{gamma}', {'gamma': gamma, 'method': 'opencv'}),
            (artwork.equalize_hist, 'eq_hist_opencv', {'method': 'opencv'}),
        ]

        logging.info(f"Обработка изображения ручными методами")
        for function, suffix, kwargs in operations_manual:
            result = function(**kwargs)
            out_path = os.path.join(self._output_dir, f'image_{prefix}_{suffix}.jpg')
            cv2.imwrite(out_path, result.image)
        logging.info(f"Обработка изображения ручными методами завершена")

        logging.info(f"Обработка изображения opencv методами")
        for function, suffix, kwargs in operations_opencv:
            result = function(**kwargs)
            out_path = os.path.join(self._output_dir, f'image_{prefix}_{suffix}.jpg')
            cv2.imwrite(out_path, result.image)
        logging.info(f"Обработка изображения opencv методами завершена")

        logging.info(f"Обработка с префиксом '{prefix}' завершена.")

    @timeit
    def process_single_image(self) -> None:
        logging.info("Запуск пайплайна обработки изображений")

        logging.info("Получение случайной картины")
        artwork_original = self.download_random_painting()
        logging.info("Получение случайной картины завершено")

        logging.info("Обработка оригинального изображения")
        self.process_artwork(artwork_original, prefix='color')
        logging.info("Обработка оригинального изображения завершена")

        logging.info("Обработка ЧБ изображения")
        artwork_grayscale = artwork_original.grayscale()
        self.process_artwork(artwork_grayscale, prefix='gray')
        logging.info("Обработка ЧБ изображения завершена")

        logging.info("Создание sobel-версии artwork")
        artwork_sobel = artwork_grayscale.sobel()
        logging.info("Создание sobel-версии artwork завершено")

        logging.info("Сложение оригинального и sobel artwork")
        artwork_sum = artwork_original + artwork_sobel
        sum_path = os.path.join(self._output_dir, 'image_original_plus_sobel.jpg')
        cv2.imwrite(sum_path, artwork_sum.image)
        logging.info(f"Результат сложения сохранен в {sum_path}")

        logging.info("Демонстрация полиморфизма: выравнивание гистограммы для разных типов")
        for a in [artwork_original, artwork_grayscale]:
            eq = a.equalize_hist(method='opencv')
            out_path = os.path.join(self._output_dir, f'image_eq_{a.__class__.__name__}.jpg')
            cv2.imwrite(out_path, eq.image)
            logging.info(f"Сохранён результат для {a.__class__.__name__} в {out_path}")

        logging.info("Пайплайн успешно завершен")


if __name__ == '__main__':
    processor = ImageProcessor()
    processor.run_pipeline()
