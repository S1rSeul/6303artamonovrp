import asyncio
from concurrent.futures import ProcessPoolExecutor

import aiofiles
import aiohttp
import shutil
from collections.abc import AsyncGenerator


import csv
import json
import logging
import os
import random
import time
from typing import List, Optional

import cv2

import numpy as np

from metetl.images.models import ColorArtwork, GrayscaleArtwork
from metetl.logging_config import configure_logging


class ImageProcessor:
    __slots__ = ('_csv_path', '_output_dir', '_download_semaphore')

    def __init__(
        self,
        csv_path: str = 'data/MetObjects.csv',
        output_dir: str = 'images',
        max_concurrent_downloads: int = 10
    ):
        self._csv_path = csv_path
        self._output_dir = output_dir
        self._download_semaphore = asyncio.Semaphore(max_concurrent_downloads)

    @staticmethod
    def _load_painting_ids(csv_path: str, count: Optional[int] = None) -> List[str]:
        random.seed(1)

        paintings = []
        with open(csv_path, mode='r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('Classification') == 'Paintings' and row.get('Is Public Domain') == 'True':
                    paintings.append(row.get("Object ID"))
        if count is None:
            return paintings
        return random.sample(paintings, min(count, len(paintings)))

    @staticmethod
    def _process_artwork_in_subprocess(task_data: tuple) -> None:
        configure_logging()

        idx, object_id, image_path, image_dir, metadata = task_data
        num = idx + 1

        try:
            logging.debug(f"Обработка изображения {num} начата (ID: {object_id})")

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
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

            logging.debug(f"Обработка изображения {num} завершена (ID: {object_id})")

        except Exception as e:
            logging.error(f"Ошибка обработки изображения {num} (ID: {object_id}): {e}", exc_info=True)

    async def _download_one_image(
            self,
            session: aiohttp.ClientSession,
            idx: int,
            object_id: str,
            output_dir: str,
            semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Optional[dict]:
        num = idx + 1

        try:
            if semaphore:
                async with semaphore:
                    return await self._do_download(session, num, object_id, output_dir)
            else:
                return await self._do_download(session, num, object_id, output_dir)
        except Exception as e:
            logging.error(f"Ошибка скачивания изображения {num} (ID: {object_id}): {e}", exc_info=True)
            return None

    @staticmethod
    async def _do_download(
            session: aiohttp.ClientSession,
            num: int,
            object_id: str,
            output_dir: str
    ) -> dict:
        logging.debug(f"Скачивание изображения {num} начато (ID: {object_id})")

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

        img_dir = os.path.join(output_dir, f"{num}_{object_id}")
        os.makedirs(img_dir, exist_ok=True)

        orig_path = os.path.join(img_dir, f"{num}_{object_id}_original.jpg")
        meta_path = os.path.join(img_dir, f"{num}_{object_id}_metadata.json")

        async with aiofiles.open(orig_path, 'wb') as f:
            await f.write(image_data)
        async with aiofiles.open(meta_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(metadata, indent=2, ensure_ascii=False))

        logging.debug(f"Скачивание изображения {num} завершено (ID: {object_id})")
        return {
            'idx': num - 1,
            'num': num,
            'object_id': object_id,
            'image_path': orig_path,
            'image_dir': img_dir,
            'metadata': metadata,
        }

    async def _get_images(self, painting_ids: list) -> AsyncGenerator[dict]:
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._download_one_image(session, idx, obj_id, self._output_dir, self._download_semaphore)
                for idx, obj_id in enumerate(painting_ids)
            ]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result is not None:
                    yield result

    async def run_pipeline(self, painting_ids: List[str]) -> None:
        if os.path.exists(self._output_dir):
            logging.debug(f"Очистка папки {self._output_dir}...")
            shutil.rmtree(self._output_dir)
        os.makedirs(self._output_dir, exist_ok=True)

        start = time.perf_counter()
        logging.info(f"Запуск пайплайна обработки {len(painting_ids)} изображений")

        for i, pid in enumerate(painting_ids):
            logging.debug(f"Изображению {i+1} присвоено ID: {pid}")

        with ProcessPoolExecutor() as executor:
            loop = asyncio.get_running_loop()
            tasks = []

            async for result in self._get_images(painting_ids):
                task_data = (
                    result['idx'],
                    result['object_id'],
                    result['image_path'],
                    result['image_dir'],
                    result['metadata'],
                )

                tasks.append(
                    loop.run_in_executor(executor, self._process_artwork_in_subprocess, task_data)
                )

            await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - start
        logging.info(f"Общее время работы: {elapsed:.2f} секунд")
        logging.info("Пайплайн успешно завершён")