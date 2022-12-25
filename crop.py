import cv2
import numpy as np

path_in = 'in/*'
path_out = 'out'
window_name = 'crop'
size_max_image = 500


def get_image_width_height(image):
    image_width = image.shape[1]  # current image's width
    image_height = image.shape[0]  # current image's height
    return image_width, image_height


def calculate_scaled_dimension(scale, image):
    # http://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
    image_width, image_height = get_image_width_height(image)
    ratio_of_new_with_to_old = scale / image_width
    dimension = (scale, int(image_height * ratio_of_new_with_to_old))
    return dimension


def rotate_image(image, degree=180):
    # http://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
    image_width, image_height = get_image_width_height(image)
    center = (image_width / 2, image_height / 2)
    M = cv2.getRotationMatrix2D(center, degree, 1.0)
    image_rotated = cv2.warpAffine(image, M, (image_width, image_height))
    return image_rotated


def scale_image(image, size):
    image_resized_scaled = cv2.resize(
        image,
        calculate_scaled_dimension(
            size,
            image
        ),
        interpolation=cv2.INTER_AREA
    )
    return image_resized_scaled


def detect_box(image, cropIt=True):
    # https://stackoverflow.com/questions/36982736/how-to-crop-biggest-rectangle-out-of-an-image/36988763
    # Transform colorspace to YUV
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_y = np.zeros(image_yuv.shape[0:2], np.uint8)
    image_y[:, :] = image_yuv[:, :, 0]

    # Blur to filter high frequency noises
    image_blurred = cv2.GaussianBlur(image_y, (3, 3), 0)

    # Apply canny edge-detector
    edges = cv2.Canny(image_blurred, 100, 300, apertureSize=3)

    # Find extrem outer contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # https://stackoverflow.com/questions/37803903/opencv-and-python-for-auto-cropping
    # Remove large countours
    new_contours = []
    for c in contours:
        if cv2.contourArea(c) < 4000000:
            new_contours.append(c)

    # Get overall bounding box
    best_box = [-1, -1, -1, -1]
    for c in new_contours:
        x, y, w, h = cv2.boundingRect(c)
        if best_box[0] < 0:
            best_box = [x, y, x + w, y + h]
        else:
            if x < best_box[0]:
                best_box[0] = x
            if y < best_box[1]:
                best_box[1] = y
            if x + w > best_box[2]:
                best_box[2] = x + w
            if y + h > best_box[3]:
                best_box[3] = y + h

    if (cropIt):
        image = image[best_box[1]:best_box[3], best_box[0]:best_box[2]]

    return image


def show_image(image, window_name):
    # Show image
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    image_width, image_height = get_image_width_height(image)
    cv2.resizeWindow(window_name, image_width, image_height)

    # Wait before closing
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cut_of_top(image, pixel):
    image_width, image_height = get_image_width_height(image)

    # startY, endY, startX, endX coordinates
    new_y = 0 + pixel
    image = image[new_y:image_height, 0:image_width]
    return image


def cut_of_bottom(image, pixel):
    image_width, image_height = get_image_width_height(image)

    # startY, endY, startX, endX coordinates
    new_height = image_height - pixel
    image = image[0:new_height, 0:image_width]
    return image


def aaa():
    image_original = cv2.imread("", cv2.IMREAD_COLOR)
    image_copy = image_original.copy()
    imgheight = image_original.shape[0]
    imgwidth = image_original.shape[1]
    print('imgheight', imgheight)
    print('imgwidth', imgwidth)

    new_height = (imgheight / 100) * 30
    new_width = (imgwidth / 100) * 30
    print('new_height', new_height)
    print('new_width', new_width)

    start_width = (imgwidth - new_width) / 2
    end_width = start_width + new_width

    start_height = (imgheight - new_height) / 2
    end_height = start_height + new_height

    cropped_image = image_original[int(start_height):int(end_height), int(start_width):int(end_width)]
    cv2.imshow("cropped", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
