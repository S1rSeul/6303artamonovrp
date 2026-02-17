import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import time
import os


def manual_grayscale(image):
    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray

def opencv_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def manual_convolve(image, kernel):
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    windows = sliding_window_view(padded, (k_h, k_w))
    result = np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))
    return np.clip(result, 0, 255).astype(np.uint8)

def manual_color_convolve(image, kernel):
    b, g, r = cv2.split(image)

    b_conv = manual_convolve(b, kernel)
    g_conv = manual_convolve(g, kernel)
    r_conv = manual_convolve(r, kernel)

    return cv2.merge([b_conv, g_conv, r_conv])

def opencv_convolve(image, kernel):
    return cv2.filter2D(image, ddepth=-1, kernel=kernel)

def gaussian_kernel(size, sigma):
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= 2 * np.pi * sigma**2
    kernel /= kernel.sum()
    return kernel.astype(np.float32)

def opencv_gaussian(image, ksize, sigma):
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

def process_image():
    output_dir = 'paintings'
    filename = 'image'
    image_path = 'paintings/image.jpg'
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    ksize = 5
    sigma = 1.0
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)

    start_manual = time.perf_counter()
    gray_manual = manual_grayscale(image)
    end_manual = time.perf_counter()
    time_manual = end_manual - start_manual
    print(f"Ручной grayscale: {time_manual:.6f} сек")

    manual_out = os.path.join(output_dir, f"{filename}_grayscale_manual.jpg")
    cv2.imwrite(manual_out, gray_manual)

    start_opencv = time.perf_counter()
    gray_opencv = opencv_grayscale(image)
    end_opencv = time.perf_counter()
    time_opencv = end_opencv - start_opencv
    print(f"OpenCV grayscale: {time_opencv:.6f} сек")

    opencv_out = os.path.join(output_dir, f"{filename}_grayscale_opencv.jpg")
    cv2.imwrite(opencv_out, gray_opencv)

    start_manual = time.perf_counter()
    convolve_manual = manual_color_convolve(image, kernel)
    end_manual = time.perf_counter()
    time_manual = end_manual - start_manual
    print(f"\nРучной convolve: {time_manual:.6f} сек")

    manual_out = os.path.join(output_dir, f"{filename}_convolve_manual.jpg")
    cv2.imwrite(manual_out, convolve_manual)

    start_opencv = time.perf_counter()
    convolve_opencv = opencv_convolve(image, kernel)
    end_opencv = time.perf_counter()
    time_opencv = end_opencv - start_opencv
    print(f"OpenCV convolve: {time_opencv:.6f} сек")

    opencv_out = os.path.join(output_dir, f"{filename}_convolve_opencv.jpg")
    cv2.imwrite(opencv_out, convolve_opencv)

    start_manual = time.perf_counter()
    kernel = gaussian_kernel(ksize, sigma)
    gaussian_manual = manual_color_convolve(image, kernel)
    end_manual = time.perf_counter()
    time_manual = end_manual - start_manual
    print(f"\nРучной gaussian: {time_manual:.6f} сек")

    manual_out = os.path.join(output_dir, f"{filename}_gaussian_manual_ks{ksize}_s{sigma}.jpg")
    cv2.imwrite(manual_out, gaussian_manual)

    start_manual = time.perf_counter()
    gaussian_opencv = opencv_gaussian(image, ksize, sigma)
    end_manual = time.perf_counter()
    time_manual = end_manual - start_manual
    print(f"OpenCV gaussian: {time_manual:.6f} сек")

    opencv_out = os.path.join(output_dir, f"{filename}_gaussian_opencv_ks{ksize}_s{sigma}.jpg")
    cv2.imwrite(opencv_out, gaussian_opencv)

if __name__ == '__main__':
    process_image()