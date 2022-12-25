import math
import os
import tempfile
from typing import Tuple, Union
import cv2
import numpy as np
from crop import detect_box
import json
from PIL.ExifTags import TAGS

try:
    from PIL import Image
except ImportError:
    import Image


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny2(image):
    return cv2.Canny(image, 100, 200)


# run cleanImg by func above
def cleanImg(image):
    grayscale1 = get_grayscale(image)
    grayscale1 = cv2.bitwise_not(grayscale1)
    thresh = thresholding(grayscale1)
    opening1 = opening(thresh)
    canny3 = canny2(opening1)
    return canny3


def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = min(1, (2480 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename


def set_image_dpi_300(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    if length_x <= 2480:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_filename = temp_file.name
        im.save(temp_filename, dpi=(300, 300))
    else:
        factor = min(1, (2480 / length_x))
        size = int(factor * length_x), int(factor * width_y)
        im_resized = im.resize(size, Image.ANTIALIAS)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_filename = temp_file.name
        im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename


def bytesto(bytes, to, bsize=1024):
    a = {'k': 1, 'm': 2, 'g': 3, 't': 4, 'p': 5, 'e': 6}
    r = float(bytes)
    return bytes / (bsize ** a[to])


def file_size(file_path):
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return bytesto(file_info.st_size, 'm')


def crop_doc(output_path):
    print('Start to crop_doc')
    image = cv2.imread(output_path)
    image = detect_box(image, True)
    # Write out file
    cv2.imwrite(output_path, image)
    print('End of crop_doc')
    return


def rotate_cv(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def validateJSON(jsonData):
    try:
        json.loads(jsonData)
    except ValueError as err:
        return False
    return True


def getDpi(fn):
    im = Image.open(fn)
    try:
        dpi = im.info['dpi']
    except:
        t = {}
        info = im._getexif()
        if not info:
            return False

        for k, v in info.items():
            tt = TAGS.get(k)
            if tt in ('XResolution', 'YResolution'):
                t[tt] = v

        dpi = [item[1] for item in sorted(t.items())]
    if not dpi:
        return False
    else:
        if isinstance(dpi[0], tuple):
            w_dpi, h_dpi = dpi[0][0], dpi[1][0]
        else:
            w_dpi, h_dpi = dpi
        return w_dpi


def get_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def biggestRectangle(contours):
    biggest = None
    max_area = 0
    indexReturn = -1
    for index in range(len(contours)):
        i = contours[index]
        area = cv2.contourArea(i)
        if area > 100:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.1 * peri, True)
            if area > max_area:  # and len(approx)==4:
                biggest = approx
                max_area = area
                indexReturn = index
    return indexReturn
