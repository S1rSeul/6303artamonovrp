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

def process_image():
    output_dir = 'paintings'
    filename = 'image'
    image_path = 'paintings/image.jpg'
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

if __name__ == '__main__':
    process_image()