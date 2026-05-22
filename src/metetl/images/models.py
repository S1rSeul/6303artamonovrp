from abc import ABC, abstractmethod

import cv2

import numpy as np
from numpy.typing import NDArray


ImageU8 = NDArray[np.uint8]
ImageF32 = NDArray[np.float32]


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

    def convolve(self, kernel: ImageF32, astype: str = 'int', method: str = 'manual') -> 'Artwork':
        result = self._convolve_array(kernel, astype, method)
        return self.__class__(result, self.metadata)

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

    def grayscale(self, method: str = 'manual') -> 'GrayscaleArtwork':
        if method == 'manual':
            weights = np.array((0.114, 0.587, 0.299), dtype=np.float32)
            gray = np.clip(self.image @ weights, 0, 255).astype(np.uint8)
        elif method == 'opencv':
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("method должен быть 'manual' или 'opencv'")
        return GrayscaleArtwork(gray, self.metadata)

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

    def grayscale(self, method: str = 'manual') -> 'GrayscaleArtwork':
        return GrayscaleArtwork(self.image, self.metadata)

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