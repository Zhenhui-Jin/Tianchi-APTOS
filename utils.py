import os
import cv2

import config
from config import TRAIN_DATA_FILE, DATA_PATH, TEST_DATA_FILE, TRAIN_DATA_FILE_NEW, TEST_DATA_FILE_NEW, \
    CST_TRAIN_DATA_FILE, CST_TEST_DATA_FILE


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


def crop_all():
    """
    裁剪
    :return:
    """
    images = []
    load_all_image_path(TRAIN_DATA_FILE, images)
    for path in images:
        write_image(os.path.join(TRAIN_DATA_FILE_NEW, os.path.split(path)[-1]), read_image(path))

    images = []
    load_all_image_path(TEST_DATA_FILE, images)
    for path in images:
        write_image(os.path.join(TEST_DATA_FILE_NEW, os.path.split(path)[-1]), read_image(path))


def processing_all():
    """
    边缘检测
    :return:
    """
    from processing import Processing

    process = Processing()

    images = []
    load_all_image_path(TRAIN_DATA_FILE_NEW, images)
    for path in images:
        print(path)
        img = process.blur_canny_cst(path)
        save_path = os.path.join(CST_TRAIN_DATA_FILE, os.path.split(path)[-1])
        cv2.imwrite(save_path, img)

    print('*' * 50)
    images = []
    load_all_image_path(TEST_DATA_FILE_NEW, images)
    for path in images:
        print(path)
        img = process.blur_canny_cst(path)
        save_path = os.path.join(CST_TEST_DATA_FILE, os.path.split(path)[-1])
        cv2.imwrite(save_path, img)


def __load_all_image_path__(self, path: str, images: list):
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
            self.__load_all_image_path__(file_path, images)
        elif file.endswith('.jpg'):
            split = file.replace('.jpg', '').split('_')
            file_id = split[0]
            if len(split) > 2:
                file_number = split[1] + split[2][-3:]
            else:
                file_number = split[1]
            # _1_000000
            images.append({'patient ID': file_id,
                           'LR': file_id[-1],
                           'ImgNumber': file_number,
                           'ImgPath': file_path.replace(config.root_path + '\\', '').replace('\\', '/')})


def before_after_to_csv():
    dataLoad = DataLoad(TRAIN_DATA_FILE_NEW, TEST_DATA_FILE_NEW)

    __load_all_image_path__(train_path, train_images)
    # __load_all_image_path__(test_path, test_images)

    train_images = pd.DataFrame(train_images, columns=['patient ID', 'LR', 'ImgNumber', 'ImgPath'])
    test_images = pd.DataFrame(test_images, columns=['patient ID', 'LR', 'ImgNumber', 'ImgPath'])

    train_data: DataFrame = pd.merge(self.get_train_csv(), self.train_images, on='patient ID')
    test_data: DataFrame = pd.merge(self.get_test_csv(), self.test_images, on='patient ID')

    number = dataLoad.train_data['ImgNumber'].apply(lambda num: int(num[0]))
    train_data1 = dataLoad.train_data[dataLoad.train_data.ImgNumber]


if __name__ == '__main__':
    crop_all()
    before_after_to_csv()
