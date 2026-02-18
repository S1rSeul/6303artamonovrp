import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import time
import os


SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

SOBEL_Y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)


def manual_grayscale(image):
    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def opencv_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def manual_convolve(image, kernel, astype = 'int'):
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    windows = sliding_window_view(padded, (k_h, k_w))
    result = np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))

    if astype == 'int':
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        return result.astype(np.float32)


def manual_color_convolve(image, kernel, astype = 'int'):
    if astype == 'int':
        b, g, r = cv2.split(image)
    else:
        b, g, r = cv2.split(image.astype(np.float32) / 255.0)

    b_conv = manual_convolve(b, kernel, astype)
    g_conv = manual_convolve(g, kernel, astype)
    r_conv = manual_convolve(r, kernel, astype)

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


def manual_magnitude(gx, gy):
    return np.sqrt(gx**2 + gy**2)


def normalize_magnitude(magnitude):
    magnitude_norm = magnitude - magnitude.min()
    if magnitude_norm.max() > 0:
        magnitude_norm = magnitude_norm / magnitude_norm.max() * 255
    return magnitude_norm.astype(np.uint8)


def manual_gamma_correction(image, gamma):
    image = image.astype(np.float32) / 255.0
    corrected = np.power(image, gamma)
    return (corrected * 255).astype(np.uint8)


def opencv_gamma_correction(image, gamma):
    image = image.astype(np.float32) / 255.0
    corrected = cv2.pow(image, gamma)
    return (corrected * 255).astype(np.uint8)


def manual_equalize_hist(image):
    hist, bins = np.histogram(image.flatten(), 256, (0, 256))

    cdf = hist.cumsum()
    cdf_norm = cdf * 255 / cdf[-1]

    lut = np.round(cdf_norm).astype(np.uint8)

    return lut[image]


def manual_equalize_hist_color(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l_eq = manual_equalize_hist(l)

    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def opencv_equalize_hist(image):
    return cv2.equalizeHist(image)


def opencv_equalize_hist_color(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l_eq = cv2.equalizeHist(l)

    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def time_and_save(function, image, out_path, description, *args, **kwargs):
    start = time.perf_counter()
    result = function(image, *args, **kwargs)
    end = time.perf_counter()
    print(f"{description}: {end - start:.6f} сек")
    if out_path is not None:
        cv2.imwrite(out_path, result)
    return result


def process_image():
    output_dir = 'paintings'
    filename = 'image'
    image_path = 'paintings/image.jpg'
    sharpen_kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    ksize = 5
    sigma = 1.0
    gamma = 0.5
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)

    time_and_save(manual_grayscale, image,
                                f"{output_dir}/{filename}_grayscale_manual.jpg",
                                "Ручной grayscale")
    time_and_save(opencv_grayscale, image,
                                f"{output_dir}/{filename}_grayscale_opencv.jpg",
                                "OpenCV grayscale")

    time_and_save(manual_color_convolve, image,
                                    f"{output_dir}/{filename}_convolve_manual.jpg",
                                    "\nРучной convolve",
                                    kernel=sharpen_kernel)

    time_and_save(opencv_convolve, image,
                                    f"{output_dir}/{filename}_convolve_opencv.jpg",
                                    "OpenCV convolve",
                                    kernel=sharpen_kernel)

    gauss_kernel = gaussian_kernel(ksize, sigma)

    time_and_save(manual_color_convolve, image,
                                    f"{output_dir}/{filename}_gaussian_manual_ks{ksize}_s{sigma}.jpg",
                                    "\nРучной gaussian",
                                    kernel=gauss_kernel)

    time_and_save(opencv_gaussian, image,
                                    f"{output_dir}/{filename}_gaussian_opencv_ks{ksize}_s{sigma}.jpg",
                                    "OpenCV gaussian",
                                    ksize=ksize, sigma=sigma)

    gx_manual = time_and_save(manual_color_convolve, image,
                              None,
                              "\nРучной sobel Gx",
                              kernel=SOBEL_X, astype='float')
    gy_manual = time_and_save(manual_color_convolve, image,
                              None,
                              "Ручной sobel Gy",
                              kernel=SOBEL_Y, astype='float')
    mag_manual = manual_magnitude(gx_manual, gy_manual)
    mag_norm = normalize_magnitude(mag_manual)
    cv2.imwrite(f"{output_dir}/{filename}_sobel_mag_manual.jpg", mag_norm)

    gx_opencv = time_and_save(cv2.Sobel, image,
                              None,
                              "OpenCV sobel Gx",
                              ddepth=cv2.CV_32F, dx=1, dy=0)
    gy_opencv = time_and_save(cv2.Sobel, image,
                              None,
                              "OpenCV sobel Gy",
                              ddepth=cv2.CV_32F, dx=0, dy=1)
    mag_opencv = cv2.magnitude(gx_opencv, gy_opencv)
    mag_norm_opencv = normalize_magnitude(mag_opencv)
    cv2.imwrite(f"{output_dir}/{filename}_sobel_mag_opencv.jpg", mag_norm_opencv)

    time_and_save(manual_gamma_correction, image,
                  f"{output_dir}/{filename}_gamma_manual_g{gamma}.jpg",
                  "\nРучная гамма-коррекция",
                  gamma=gamma)

    time_and_save(opencv_gamma_correction, image,
                  f"{output_dir}/{filename}_gamma_opencv_g{gamma}.jpg",
                  "OpenCV гамма-коррекция",
                  gamma=gamma)

    time_and_save(manual_equalize_hist_color, image,
                  f"{output_dir}/{filename}_eq_hist_manual.jpg",
                  "\nРучное выравнивание гистограммы")

    time_and_save(opencv_equalize_hist_color, image,
                  f"{output_dir}/{filename}_eq_hist_opencv.jpg",
                  "OpenCV выравнивание гистограммы")


if __name__ == '__main__':
    process_image()