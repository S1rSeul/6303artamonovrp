import csv
import json
import os
import random
import time
from typing import Any, Callable, Optional

import cv2


import numpy as np
import requests
from numpy.typing import NDArray


ImageU8 = NDArray[np.uint8]
ImageF32 = NDArray[np.float32]


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
        return Artwork(self._image + other._image, self._metadata)

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