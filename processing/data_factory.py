import os

import cv2
import pandas as pd

import config


def _all_img_path_source():
    images = []
    _all_img_path_preliminary(config.SOURCE_TEST_IMG_PATH_PRELIMINARY, images, 'test')
    _all_img_path_preliminary(config.SOURCE_TRAIN_IMG_PATH_PRELIMINARY, images, 'train')
    _all_img_path_final(config.SOURCE_TEST_IMG_PATH_FINAL, images, 'test')
    _all_img_path_final(config.SOURCE_TRAIN_IMG_PATH_FINAL, images, 'train')
    return images


def _all_img_path_final(path: str, images: list, data_type: str):
    """
    加载所有图像路径
    :param path:路径
    :param images:
    :param data_type: test or train
    :return:
    """
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            _all_img_path_final(file_path, images, data_type)
        elif file.endswith('.jpg') or file.endswith('.png'):
            source_path = file_path.replace('\\', '/')
            path_split = source_path.split('/')
            file_id = path_split[-3]

            if file_id.endswith('L'):
                L0R1 = 0
            else:
                L0R1 = 1

            if path_split[-2].lower().startswith('post'):
                injection = 'Post injection'
                after = 1
            else:
                injection = 'Pre injection'
                after = 0

            final = 1
            name = f'{file_id}_{final}_{after}_{L0R1}_{file}'

            if data_type == 'test':
                processed_path = config.PROCESSED_TEST_IMG_PATH
            else:
                processed_path = config.PROCESSED_TRAIN_IMG_PATH
            processed_path = os.path.join(processed_path, name).replace('\\', '/')

            image_name = file.replace('.jpg', '').replace('.png', '')

            images.append({'patient ID': file_id,
                           'L0R1': L0R1,
                           'final': final,
                           'after': after,
                           'injection': injection,
                           'image name': image_name,
                           'processed_path': processed_path,
                           'source_path': source_path,
                           'data_type': data_type})


def _all_img_path_preliminary(path: str, images: list, data_type: str):
    """
    加载所有图像路径
    :param path:路径
    :param images:
    :param data_type: test or train
    :return:
    """
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            _all_img_path_preliminary(file_path, images, data_type)
        elif file.endswith('.jpg') or file.endswith('.png'):

            if file.endswith('.jpg'):
                suffix = '.jpg'
            else:
                suffix = '.png'

            split = file.replace('.jpg', '').replace('.png', '').split('_')
            file_id = split[0]
            if len(split) > 2:
                # _1_000000
                file_number = split[1] + split[2]
            else:
                file_number = split[1]

            if file_number.startswith('1'):
                injection = 'Post injection'
                after = 1
            else:
                injection = 'Pre injection'
                after = 0

            if file_id.endswith('L'):
                L0R1 = 0
            else:
                L0R1 = 1

            final = 1
            name = f'{file_id}_{final}_{after}_{L0R1}_{file_number}_{suffix}'

            if data_type == 'test':
                processed_path = config.PROCESSED_TEST_IMG_PATH
            else:
                processed_path = config.PROCESSED_TRAIN_IMG_PATH
            processed_path = os.path.join(processed_path, name).replace('\\', '/')

            image_name = file.replace('.jpg', '').replace('.png', '')

            images.append({'patient ID': file_id,
                           'L0R1': L0R1,
                           'final': 0,
                           'after': after,
                           'injection': injection,
                           'image name': image_name,
                           'processed_path': processed_path,
                           'source_path': file_path.replace('\\', '/'),
                           'data_type': data_type})


def read_image(path: str):
    """
    使用OpenCV加载图像数据
    :param path:
    :return:
    """
    img = cv2.imread(path)
    # 裁剪
    img = img[50:450, 550:1200, :]
    return img


def write_image(path: str, img):
    cv2.imwrite(path, img)
    return img


def crop_img():
    images = _all_img_path_source()
    print('crop_img')
    for image in images:
        print(image['image name'])
        processed_path = image['processed_path']
        source_path = image['source_path']
        write_image(processed_path, read_image(source_path))
    data = pd.DataFrame(images)
    data.to_csv(config.PROCESSED_IMAGE_CSV_PATH, index=False)


def processing_data_source_final():
    print('processing_data_source_preliminary')
    image_data = pd.read_csv(config.PROCESSED_IMAGE_CSV_PATH)
    train_pre_image_data = image_data.loc[
        (image_data['data_type'] == 'train') & (image_data['after'] == 0) & (image_data['final'] == 1)]
    train_post_image_data = image_data.loc[
        (image_data['data_type'] == 'train') & (image_data['after'] == 1) & (image_data['final'] == 1)]

    train_case = pd.read_csv(config.SOURCE_TRAIN_CASE_CSV_PATH_FINAL)
    train_pic = pd.read_csv(config.SOURCE_TRAIN_PIC_CSV_PATH_FINAL)


def processing_data_source_preliminary():
    print('processing_data_source_preliminary')
    image_data = pd.read_csv(config.PROCESSED_IMAGE_CSV_PATH)
    train_pre_image_data = image_data.loc[
        (image_data['data_type'] == 'train') & (image_data['after'] == 0) & (image_data['final'] == 0)]
    train_post_image_data = image_data.loc[
        (image_data['data_type'] == 'train') & (image_data['after'] == 1) & (image_data['final'] == 0)]

    train_preliminary = pd.read_csv(config.SOURCE_TRAIN_CSV_PATH_PRELIMINARY)

    for column in ['preVA', 'preCST', 'preIRF', 'preSRF', 'prePED', 'preHRF',
                   'VA', 'continue injection', 'CST', 'IRF', 'SRF', 'PED', 'HRF']:
        train_preliminary.loc[train_preliminary[column].isna(), column] = train_preliminary[column].mean()

    train_case = train_preliminary[
        ['patient ID', 'gender', 'age', 'diagnosis', 'anti-VEGF', 'preVA', 'preCST', 'VA', 'CST',
         'continue injection']]
    train_pre = train_preliminary[['patient ID', 'preIRF', 'preSRF', 'prePED', 'preHRF']].copy()
    train_pre.rename(columns={'preIRF': 'IRF', 'preSRF': 'SRF', 'prePED': 'PED', 'preHRF': 'HRF'}, inplace=True)
    train_pre = train_pre.merge(train_case, on='patient ID', sort=True)
    train_pre = train_pre_image_data.merge(train_pre, on='patient ID', sort=True)

    train_post = train_preliminary[['patient ID', 'IRF', 'SRF', 'PED', 'HRF']].copy()
    train_post = train_post.merge(train_case, on='patient ID', sort=True)
    train_post = train_post_image_data.merge(train_post, on='patient ID', sort=True)

    train = pd.concat([train_pre, train_post], sort=True)
    train = train[['patient ID', 'gender', 'age', 'diagnosis', 'anti-VEGF', 'preVA', 'VA', 'preCST', 'CST',
                   'IRF', 'SRF', 'PED', 'HRF', 'continue injection', 'L0R1', 'injection', 'image name', 'after',
                   'final', 'data_type', 'processed_path', 'source_path']]
    train.to_csv(config.PROCESSED_TRAIN_CSV_PATH_PRELIMINARY, index=False)
