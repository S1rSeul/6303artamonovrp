import csv
import json
import os
import random
import time
from typing import Any, Callable

import cv2

import numpy as np
from numpy.typing import NDArray

import requests


ImageU8 = NDArray[np.uint8]
ImageF32 = NDArray[np.float32]


def timeit(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[TIME] {func.__name__} выполнена за {end - start:.6f} секунд")
        return result
    return wrapper


def get_painting_id(csv_path: str) -> str:
    paintings = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if (row.get('Classification') == 'Paintings'
                    and row.get('Is Public Domain') == 'True'):
                paintings.append(row.get("Object ID"))

    return random.choice(paintings)


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


class Artwork:
    __slots__ = ('_image', '_metadata')

    def __init__(self, image: ImageU8, metadata: dict):
        self._image = image
        self._metadata = metadata

    @property
    def image(self) -> ImageU8:
        return self._image

    @property
    def metadata(self) -> dict:
        return self._metadata

    def __str__(self) -> str:
        title = self._metadata.get('Title', 'Неизвестен')
        artist = self._metadata.get('Artist Display Name', 'Неизвестен')
        return f"Artwork: '{title}' by {artist}"

    def __add__(self, other: 'Artwork') -> 'Artwork':
        if not isinstance(other, Artwork):
            raise TypeError("Можно складывать только объекты Artwork")

        if self._metadata != other._metadata:
            raise ValueError("Можно складывать только изображения с одинаковыми метаданными")

        new_image = cv2.add(self._image, other._image)
        return Artwork(new_image, self._metadata)

    def grayscale(self, method: str = 'manual') -> ImageU8:
        if method == 'manual':
            weights = np.array((0.114, 0.587, 0.299), dtype=np.float32)
            return np.clip(self._image @ weights, 0, 255).astype(np.uint8)
        elif method == 'opencv':
            return cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("method должен быть 'manual' или 'opencv'")

    def convolve(self, kernel: ImageF32, astype: str = 'int', method: str = 'manual') -> ImageU8 | ImageF32:
        if method == 'manual':
            k_h, k_w = kernel.shape
            pad_h, pad_w = k_h // 2, k_w // 2

            padded_width = [(pad_h, pad_h), (pad_w, pad_w)]
            if self._image.ndim == 3:
                padded_width.append((0, 0))

            padded = np.pad(self._image, padded_width, mode='reflect')

            windows = np.lib.stride_tricks.sliding_window_view(padded, (k_h, k_w), axis=(0, 1))
            result = np.tensordot(windows, kernel, axes=((-2, -1), (0, 1)))

            if astype == 'int':
                return np.clip(result, 0, 255).astype(np.uint8)
            else:
                return result.astype(np.float32)
        elif method == 'opencv':
            return cv2.filter2D(self._image, ddepth=-1, kernel=kernel)
        else:
            raise ValueError("method должен быть 'manual' или 'opencv'")

    def gaussian(self, ksize: int, sigma: float, method: str = 'manual') -> ImageU8:
        if method == 'manual':
            k = ksize // 2
            x, y = np.mgrid[-k:k + 1, -k:k + 1]
            kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            kernel /= kernel.sum()
            return self.convolve(kernel)
        elif method == 'opencv':
            return cv2.GaussianBlur(self._image, (ksize, ksize), sigma)
        else:
            raise ValueError("method должен быть 'manual' или 'opencv'")

    def sobel(self, method: str = 'manual') -> ImageU8:
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

            gx = self.convolve(sobel_x, 'float')
            gy = self.convolve(sobel_y, 'float')

            magnitude = np.sqrt(gx ** 2 + gy ** 2)

            return magnitude.astype(np.uint8)
        elif method == 'opencv':
            gx = cv2.Sobel(self._image, ddepth=cv2.CV_32F, dx=1, dy=0)
            gy = cv2.Sobel(self._image, ddepth=cv2.CV_32F, dx=0, dy=1)

            magnitude = cv2.magnitude(gx, gy)

            return magnitude.astype(np.uint8)
        else:
            raise ValueError("method должен быть 'manual' или 'opencv'")

    def gamma_correction(self, gamma: float, method: str = 'manual') -> ImageU8:
        if method == 'manual':
            image = self._image.astype(np.float32) / 255.0
            corrected = np.power(image, 1 / gamma)
            return (corrected * 255).astype(np.uint8)
        elif method == 'opencv':
            image = self._image.astype(np.float32) / 255.0
            corrected = cv2.pow(image, 1 / gamma)
            return (corrected * 255).astype(np.uint8)
        else:
            raise ValueError("method должен быть 'manual' или 'opencv'")

    @staticmethod
    def _equalize_hist_impl(image: ImageU8, method: str = 'manual') -> ImageU8:
        if method == 'manual':
            hist = np.histogram(image.flatten(), 256, (0, 256))[0]
            cdf = hist.cumsum()
            cdf_norm = cdf * 255 / cdf[-1]
            lut = np.round(cdf_norm).astype(np.uint8)
            return lut[image]
        elif method == 'opencv':
            return cv2.equalizeHist(image)
        else:
            raise ValueError("method должен быть 'manual' или 'opencv'")

    def equalize_hist(self, method: str = 'manual') -> ImageU8:
        if self._image.ndim == 2:
            return self._equalize_hist_impl(self._image, method)
        elif self._image.ndim == 3:
            if method == 'manual':
                lab = cv2.cvtColor(self._image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l_eq = self._equalize_hist_impl(l)
            elif method == 'opencv':
                lab = cv2.cvtColor(self._image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l_eq = cv2.equalizeHist(l)
            else:
                raise ValueError("method должен быть 'manual' или 'opencv'")

            lab_eq = cv2.merge([l_eq, a, b])
            return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        else:
            raise ValueError(f"Не поддерживаемая размерность: {self._image.ndim}. Ожидается 2 (ЧБ) или 3 (цветное).")


class ImageProcessor:
    __slots__ = ('_csv_path', '_output_dir')

    def __init__(self, csv_path: str = 'MetObjects.csv', output_dir: str = 'paintings'):
        self._csv_path = csv_path
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)

    @staticmethod
    def _log(message: str) -> None:
        print(f"[IMAGE PROCESSOR] {message}")

    @timeit
    def download_random_painting(self) -> Artwork:
        object_id = get_painting_id(self._csv_path)
        self._log(f"Выбрана картина ID {object_id}")
        self._log(f"Загрузка метаданных для объекта {object_id}")
        metadata = fetch_object_metadata(object_id)

        primary_image = metadata.get('primaryImage')
        if not primary_image:
            raise ValueError(f"У объекта {object_id} отсутствует primaryImage")

        img_path = os.path.join(self._output_dir, 'image.jpg')
        self._log(f"Скачивание изображения: {primary_image}")
        download_image(primary_image, img_path)
        self._log(f"Изображение сохранено в {img_path}")

        image = cv2.imread(img_path)

        json_path = os.path.join(self._output_dir, 'image.json')
        save_metadata(metadata, json_path)
        self._log(f"Метаданные сохранены в {json_path}")

        artwork = Artwork(image, metadata)
        self._log(f"Создан объект: {artwork}")
        return artwork

    @timeit
    def process_artwork(self, artwork: Artwork, prefix: str = '') -> None:
        self._log(f"Начало обработки изображения с префиксом '{prefix}'...")

        orig_path = os.path.join(self._output_dir, f'image_{prefix}_original.jpg')
        cv2.imwrite(orig_path, artwork.image)
        self._log(f"Оригинал изображения сохранен в {orig_path}")

        sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ], dtype=np.float32)
        ksize = 3
        sigma = 1.0
        gamma = 0.5

        def time_and_save(func: Callable, suffix: str, description: str, **kwargs) -> None:
            start = time.perf_counter()
            result = func(**kwargs)
            end = time.perf_counter()
            print(f"[TIME] {description}: {end - start:.6f} секунд")
            out_path = os.path.join(self._output_dir, f'image_{prefix}_{suffix}.jpg')
            cv2.imwrite(out_path, result)

        time_and_save(
            lambda: artwork.grayscale(method='manual'),
            'grayscale_manual',
            "Ручной grayscale",
        )
        time_and_save(
            lambda: artwork.grayscale(method='opencv'),
            'grayscale_opencv',
            "OpenCV grayscale",
        )

        time_and_save(
            lambda: artwork.convolve(kernel=sharpen_kernel, method='manual'),
            'convolve_manual',
            "Ручной convolve",
        )
        time_and_save(
            lambda: artwork.convolve(kernel=sharpen_kernel, method='opencv'),
            'convolve_opencv',
            "OpenCV convolve",
        )

        time_and_save(
            lambda: artwork.gaussian(ksize=ksize, sigma=sigma, method='manual'),
            f'gaussian_manual_ks{ksize}_s{sigma}',
            "Ручной gaussian",
        )
        time_and_save(
            lambda: artwork.gaussian(ksize=ksize, sigma=sigma, method='opencv'),
            f'gaussian_opencv_ks{ksize}_s{sigma}',
            "OpenCV gaussian",
        )

        time_and_save(
            lambda: artwork.sobel(method='manual'),
            'sobel_mag_manual',
            "Ручной sobel",
        )
        time_and_save(
            lambda: artwork.sobel(method='opencv'),
            'sobel_mag_opencv',
            "OpenCV sobel",
        )

        time_and_save(
            lambda: artwork.gamma_correction(gamma=gamma, method='manual'),
            f'gamma_manual_g{gamma}',
            "Ручная гамма-коррекция",
        )
        time_and_save(
            lambda: artwork.gamma_correction(gamma=gamma, method='opencv'),
            f'gamma_opencv_g{gamma}',
            "OpenCV гамма-коррекция",
        )

        time_and_save(
            lambda: artwork.equalize_hist(method='manual'),
            'eq_hist_manual',
            "Ручное выравнивание гистограммы",
        )
        time_and_save(
            lambda: artwork.equalize_hist(method='opencv'),
            'eq_hist_opencv',
            "OpenCV выравнивание гистограммы",
        )

        self._log(f"Обработка с префиксом '{prefix}' завершена.")

    def run_pipeline(self) -> None:
        self._log("Запуск пайплайна обработки изображений")

        self._log("Обработка оригинального изображения")
        artwork_original = self.download_random_painting()
        self.process_artwork(artwork_original, prefix='color')
        self._log("Обработка оригинального изображения завершена")

        self._log("Обработка ЧБ изображения")
        image_grayscale = artwork_original.grayscale(method='manual')
        image_grayscale_3c = cv2.cvtColor(image_grayscale, cv2.COLOR_GRAY2BGR)
        artwork_grayscale = Artwork(image_grayscale_3c, artwork_original.metadata.copy())
        self.process_artwork(artwork_grayscale, prefix='gray')
        self._log("Обработка ЧБ изображения завершена")

        self._log("Создание sobel-версии artwork")
        image_sobel = artwork_grayscale.sobel(method='manual')
        artwork_sobel = Artwork(image_sobel, artwork_original.metadata.copy())
        self._log("Создание sobel-версии artwork завершено")

        self._log("Сложение оригинального и sobel artwork")
        artwork_sum = artwork_original + artwork_sobel
        sum_path = os.path.join(self._output_dir, 'image_original_plus_sobel.jpg')
        cv2.imwrite(sum_path, artwork_sum.image)
        self._log(f"Результат сложения сохранен в {sum_path}")

        self._log("Пайплайн успешно завершен")


if __name__ == '__main__':
    processor = ImageProcessor()
    processor.run_pipeline()
