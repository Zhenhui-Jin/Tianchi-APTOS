import os

import cv2
import pandas as pd

import config
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
        elif file.endswith('.jpg') or file.endswith('.png'):
            split = file.replace('.jpg', '').replace('.png', '').split('_')
            file_id = split[0]
            if len(split) > 2:
                # _1_000000
                file_number = split[1] + split[2]
            else:
                file_number = split[1]
            if file_id[-1] == 'L':
                L0R1 = 0
            else:
                L0R1 = 1
            images.append({'patient ID': file_id,
                           'L0R1': L0R1,
                           'ImgNumber': file_number,
                           'ImgPath': file_path.replace(config.root_path + '\\', '').replace('\\', '/')})


def read_image(path: str):
    """
    使用OpenCV加载图像数据
    :param path:
    :return:
    """
    img = cv2.imread(path)
    # 裁剪
    img = img[:500, 500:1264, :]
    return img


def write_image(path: str, img):
    cv2.imwrite(path, img)
    return img


def crop_all(train_path, test_path, new_train_path, new_test_path):
    """
    裁剪图像
    :param train_path: 裁剪前路径
    :param test_path: 裁剪前路径
    :param new_train_path: 裁剪后路径
    :param new_test_path: 裁剪后路径
    :return:
    """
    images = []
    load_all_image_path(train_path, images)
    for image in images:
        path = image['ImgPath']
        write_image(os.path.join(new_train_path, os.path.split(path)[-1]), read_image(path))

    images = []
    load_all_image_path(test_path, images)
    for image in images:
        path = image['ImgPath']
        write_image(os.path.join(new_test_path, os.path.split(path)[-1]), read_image(path))


def load_csv(path) -> pd.DataFrame:
    """
    加载CSV数据集
    :param path: 路径
    :return: DataFrame
    """
    files = os.listdir(path)
    for file in files:
        if file.endswith('.csv'):
            data = pd.read_csv(os.path.join(path, file), header=0)
            return data


def before_after_to_csv(train_path, test_path, save_path):
    """
    数据集分成前后两部分
    :param train_path:
    :param test_path:
    :param save_path:
    :return:
    """
    train_images = []
    load_all_image_path(train_path, train_images)

    test_images = []
    load_all_image_path(test_path, test_images)

    train_images = pd.DataFrame(train_images, columns=['patient ID', 'L0R1', 'ImgNumber', 'ImgPath'])
    test_images = pd.DataFrame(test_images, columns=['patient ID', 'L0R1', 'ImgNumber', 'ImgPath'])

    train_data: pd.DataFrame = pd.merge(load_csv(train_path), train_images, on='patient ID')
    test_data: pd.DataFrame = pd.merge(load_csv(test_path), test_images, on='patient ID')

    number = train_data['ImgNumber'].apply(lambda num: int(num[0]))
    train_data_before = train_data[number == 1]
    train_data_after = train_data[number == 2]

    train_data_before.to_csv(os.path.join(save_path, 'train_data_before.csv'), index=False)
    train_data_after.to_csv(os.path.join(save_path, 'train_data_after.csv'), index=False)

    number = test_data['ImgNumber'].apply(lambda num: int(num[0]))
    test_data_before = test_data[number == 1]
    test_data_after = test_data[number == 2]

    test_data_before.to_csv(os.path.join(save_path, 'test_data_before.csv'), index=False)
    test_data_after.to_csv(os.path.join(save_path, 'test_data_after.csv'), index=False)


if __name__ == '__main__':
    crop_all(TRAIN_DATA_FILE, TEST_DATA_FILE, TRAIN_DATA_FILE_NEW, TEST_DATA_FILE_NEW)
    before_after_to_csv(TRAIN_DATA_FILE_NEW, TEST_DATA_FILE_NEW, DATA_PATH)
