import os
import cv2

from config import TRAIN_DATA_FILE, DATA_PATH, TEST_DATA_FILE, TRAIN_DATA_FILE_NEW, TEST_DATA_FILE_NEW


def load_all_image_path(path: str, images: list):
    """
    加载所有图像路径
    :param path:路径
    :param images:
    :return:
    """
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            load_all_image_path(file_path, images)
        elif file.endswith('.jpg'):
            images.append(file_path)


def read_image(path: str):
    """
    使用OpenCV加载图像数据
    :param path:
    :return:
    """
    img = cv2.imread(path)
    # 裁剪
    img = img[:500, 500:, :]
    return img


def write_image(path: str, img):
    cv2.imwrite(path, img)
    return img


if __name__ == '__main__':
    # 剪切图片另存
    images = []
    load_all_image_path(TRAIN_DATA_FILE, images)
    for path in images:
        write_image(os.path.join(TRAIN_DATA_FILE_NEW, os.path.split(path)[-1]), read_image(path))

    load_all_image_path(TEST_DATA_FILE, images)
    for path in images:
        write_image(os.path.join(TEST_DATA_FILE_NEW, os.path.split(path)[-1]), read_image(path))
