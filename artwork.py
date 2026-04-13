import asyncio
import sys
from concurrent.futures import ProcessPoolExecutor
import aiofiles
import aiohttp


import csv
import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, List

import cv2

import numpy as np
from numpy.typing import NDArray


ImageU8 = NDArray[np.uint8]
ImageF32 = NDArray[np.float32]


logging.basicConfig(level=logging.INFO, format='%(asctime)s - PID %(process)d - %(levelname)s - %(message)s')


def timeit(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logging.info(f"[TIME] {func.__name__} выполнена за {end - start:.6f} секунд")
        return result
    return wrapper


def _load_painting_ids(csv_path: str, count: int) -> List[str]:
    random.seed(1)

    paintings = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('Classification') == 'Paintings' and row.get('Is Public Domain') == 'True':
                paintings.append(row.get("Object ID"))
    return random.sample(paintings, count)


async def _download_one_image(
    session: aiohttp.ClientSession,
    idx: int,
    object_id: str,
    output_dir: str
) -> dict:
    num = idx + 1

    img_dir = os.path.join(output_dir, f"{num}_{object_id}")
    os.makedirs(img_dir, exist_ok=True)

    logging.info(f"Скачивание изображения {num} начато (ID: {object_id})")

    meta_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
    async with session.get(meta_url) as response:
        response.raise_for_status()
        metadata = await response.json()

    primary_image = metadata.get('primaryImage')
    if not primary_image:
        raise ValueError(f"У объекта {object_id} отсутствует primaryImage")

    async with session.get(primary_image) as response:
        response.raise_for_status()
        image_data = await response.read()

    orig_path = os.path.join(img_dir, f"{num}_{object_id}_original.jpg")
    meta_path = os.path.join(img_dir, f"{num}_{object_id}_metadata.json")

    async with aiofiles.open(orig_path, 'wb') as f:
        await f.write(image_data)
    async with aiofiles.open(meta_path, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(metadata, indent=2, ensure_ascii=False))

    logging.info(f"Скачивание изображения {num} завершено (ID: {object_id})")
    return {
        'idx': idx,
        'num': num,
        'object_id': object_id,
        'image_path': orig_path,
        'image_dir': img_dir,
        'metadata': metadata,
    }

def _process_artwork_in_subprocess(task_data: tuple) -> None:
    idx, object_id, image_path, image_dir = task_data
    num = idx + 1
    pid = os.getpid()
    logging.info(f"Обработка изображения {num} начата (PID {pid}, ID: {object_id})")

    meta_path = os.path.join(image_dir, f"{num}_{object_id}_metadata.json")
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение {image_path}")

    if image.ndim == 3:
        artwork = ColorArtwork(image, metadata)
    elif image.ndim == 2:
        artwork = GrayscaleArtwork(image, metadata)
    else:
        raise ValueError(f"Неподдерживаемая размерность: {image.shape}")

    base_prefix = f"{num}_{object_id}"

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

    for function, suffix, kwargs in operations_manual:
        result = function(**kwargs)
        out_path = os.path.join(image_dir, f"{base_prefix}_{suffix}.jpg")
        cv2.imwrite(out_path, result.image)

    for function, suffix, kwargs in operations_opencv:
        result = function(**kwargs)
        out_path = os.path.join(image_dir, f"{base_prefix}_{suffix}.jpg")
        cv2.imwrite(out_path, result.image)

    artwork_grayscale = artwork.grayscale()
    gray_orig_path = os.path.join(image_dir, f"{base_prefix}_grayscale_original.jpg")
    cv2.imwrite(gray_orig_path, artwork_grayscale.image)

    artwork_sobel = artwork_grayscale.sobel()
    sum_path = os.path.join(image_dir, f"{base_prefix}_original_plus_sobel.jpg")
    try:
        artwork_sum = artwork + artwork_sobel
        cv2.imwrite(sum_path, artwork_sum.image)
    except Exception as e:
        logging.error(f"Ошибка при сложении для {num}: {e}")

    for art in [artwork, artwork_grayscale]:
        eq = art.equalize_hist(method='opencv')
        eq_path = os.path.join(image_dir, f"{base_prefix}_eq_{art.__class__.__name__}.jpg")
        cv2.imwrite(eq_path, eq.image)

    logging.info(f"Обработка изображения {num} завершена (PID {pid}, ID: {object_id})")


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

    async def download_paintings_async(self, painting_ids: List[str]) -> List[dict]:
        async with aiohttp.ClientSession() as session:
            tasks = [
                _download_one_image(session, idx, obj_id, self._output_dir)
                for idx, obj_id in enumerate(painting_ids)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    logging.error(f"Ошибка при скачивании: {res}")
                    raise res
        return results

    @staticmethod
    def process_paintings(download_results: List[dict]) -> None:
        tasks = []
        for res in download_results:
            tasks.append((
                res['idx'],
                res['object_id'],
                res['image_path'],
                res['image_dir'],
            ))

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(_process_artwork_in_subprocess, task) for task in tasks]
            for future in futures:
                future.result()

    async def run_pipeline(self, num_paintings: int) -> None:
        start = time.perf_counter()
        logging.info(f"Запуск пайплайна обработки {num_paintings} изображений")

        painting_ids = _load_painting_ids(self._csv_path, num_paintings)
        for i, pid in enumerate(painting_ids):
            logging.info(f"Изображению {i+1} присвоено ID: {pid}")

        logging.info("Начало асинхронного скачивания...")
        download_start = time.perf_counter()
        download_results = await self.download_paintings_async(painting_ids)
        download_time = time.perf_counter() - download_start
        logging.info(f"Скачивание завершено за {download_time:.2f} секунд")

        logging.info(f"Запуск параллельной обработки {num_paintings} изображений...")
        proc_start = time.perf_counter()
        self.process_paintings(download_results)
        proc_time = time.perf_counter() - proc_start
        logging.info(f"Обработка завершена за {proc_time:.2f} секунд")

        complete_time = time.perf_counter() - start
        logging.info(f"Общее время работы: {complete_time:.2f} секунд")
        logging.info("Пайплайн успешно завершён")


def main():
    if len(sys.argv) == 2:
        try:
            num_images = int(sys.argv[1])
        except ValueError:
            print("Аргумент должен быть целым числом.")
            sys.exit(1)
    else:
        print("Использование: python artwork.py <количество_изображений>")
        sys.exit(1)

    processor = ImageProcessor()
    asyncio.run(processor.run_pipeline(num_images))


if __name__ == '__main__':
    main()
