import os

import cv2
import matplotlib.pyplot as plt
import pandas as pd

import config


def _all_img_path_source():
    images = []
    _all_img_path_final(config.SOURCE_TEST_IMG_PATH_FINAL, images, 'test')
    print('final test', len(images))
    _all_img_path_final(config.SOURCE_TRAIN_IMG_PATH_FINAL, images, 'train')
    print('final train', len(images))
    _all_img_path_preliminary(config.SOURCE_TEST_IMG_PATH_PRELIMINARY, images, 'test')
    _all_img_path_preliminary(config.SOURCE_TRAIN_IMG_PATH_PRELIMINARY, images, 'train')
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
        if file == '0000-1270R_2.jpg' or file == '0000-1186R_2.jpg':
            # 异常图片不处理
            continue
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
                injection = 'Pre injection'
                after = 0
            else:
                injection = 'Post injection'
                after = 1

            if file_id.endswith('L'):
                L0R1 = 0
            else:
                L0R1 = 1

            final = 1
            name = f'{file_id}_{final}_{after}_{L0R1}_{file_number}{suffix}'

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
    if img.shape[1] != 650:
        img = cv2.copyMakeBorder(img, 0, 0, 0, 650 - img.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img


def write_image(path: str, img):
    cv2.imwrite(path, img)
    return img


def crop_img():
    """
    裁剪图片
    :return:
    """
    images = _all_img_path_source()
    print('crop_img')
    for image in images:
        processed_path = image['processed_path']
        source_path = image['source_path']
        write_image(processed_path, read_image(source_path))
    data = pd.DataFrame(images)
    data.to_csv(config.PROCESSED_IMAGE_CSV_PATH, index=False)


def _processing_data_source_final_train():
    """
    处理复赛数据
    :return:
    """
    print('_processing_data_source_final_train')
    image_data = pd.read_csv(config.PROCESSED_IMAGE_CSV_PATH)
    train_pre_image_data = image_data.loc[
        (image_data['data_type'] == 'train') & (image_data['after'] == 0) & (image_data['final'] == 1)]
    train_post_image_data = image_data.loc[
        (image_data['data_type'] == 'train') & (image_data['after'] == 1) & (image_data['final'] == 1)]

    train_case = pd.read_csv(config.SOURCE_TRAIN_CASE_CSV_PATH_FINAL)

    train_case.loc[train_case['gender'] == 'Male', 'gender'] = 1
    train_case.loc[train_case['gender'] == 'Male ', 'gender'] = 1
    train_case.loc[train_case['gender'] == 'Female', 'gender'] = 2

    train_case.loc[train_case['diagnosis'] == 'CNVM', 'diagnosis'] = 1
    train_case.loc[train_case['diagnosis'] == 'PCV', 'diagnosis'] = 2
    train_case.loc[train_case['diagnosis'] == 'DME', 'diagnosis'] = 3
    train_case.loc[train_case['diagnosis'] == 'RVO', 'diagnosis'] = 4
    train_case.loc[train_case['diagnosis'] == 'CME', 'diagnosis'] = 5

    train_case.loc[train_case['anti-VEGF'] == 'Avastin', 'anti-VEGF'] = 1
    train_case.loc[train_case['anti-VEGF'] == 'Razumab', 'anti-VEGF'] = 2
    train_case.loc[train_case['anti-VEGF'] == 'Accentrix', 'anti-VEGF'] = 2
    train_case.loc[train_case['anti-VEGF'] == 'Eylea', 'anti-VEGF'] = 3
    train_case.loc[train_case['anti-VEGF'] == 'Tricort', 'anti-VEGF'] = 5
    train_case.loc[train_case['anti-VEGF'] == 'Ozrudex', 'anti-VEGF'] = 10
    train_case.loc[train_case['anti-VEGF'] == 'Ozurdex', 'anti-VEGF'] = 10
    train_case.loc[train_case['anti-VEGF'] == 'Pagenax', 'anti-VEGF'] = 0

    train_case.loc[train_case['preVA'] == 'NLP', 'preVA'] = 1

    train_final = pd.read_csv(config.SOURCE_TRAIN_PIC_CSV_PATH_FINAL)
    train_final = train_final.merge(train_case, on='patient ID', sort=True)

    train_pre = train_final[['patient ID', 'injection', 'image name', 'preVA', 'VA', 'preCST', 'CST', 'IRF', 'SRF',
                             'PED', 'HRF', 'anti-VEGF', 'gender', 'age', 'continue injection', 'diagnosis']].copy()
    # train_pre.rename(columns={'preCST': 'CST'}, inplace=True)
    train_pre = train_pre_image_data.merge(train_pre, on=['patient ID', 'injection', 'image name'], sort=True)

    train_post = train_final[['patient ID', 'injection', 'image name', 'preVA', 'VA', 'preCST', 'CST', 'IRF', 'SRF',
                              'PED', 'HRF', 'anti-VEGF', 'gender', 'age', 'continue injection', 'diagnosis']].copy()
    train_post = train_post_image_data.merge(train_post, on=['patient ID', 'injection', 'image name'], sort=True)

    train_final = pd.concat([train_pre, train_post], sort=True)
    train_final = train_final[['patient ID', 'gender', 'age', 'diagnosis', 'anti-VEGF', 'preVA', 'VA', 'preCST', 'CST',
                               'IRF', 'SRF', 'PED', 'HRF', 'continue injection', 'L0R1', 'injection', 'image name',
                               'after', 'final', 'data_type', 'processed_path', 'source_path']]
    return train_final


def _processing_data_source_preliminary_train():
    """
    处理初赛数据
    :return:
    """
    print('_processing_data_source_preliminary_train')
    image_data = pd.read_csv(config.PROCESSED_IMAGE_CSV_PATH)
    train_pre_image_data = image_data.loc[
        (image_data['data_type'] == 'train') & (image_data['after'] == 0) & (image_data['final'] == 0)]
    train_post_image_data = image_data.loc[
        (image_data['data_type'] == 'train') & (image_data['after'] == 1) & (image_data['final'] == 0)]

    train_preliminary = pd.read_csv(config.SOURCE_TRAIN_CSV_PATH_PRELIMINARY)

    for column in ['preVA', 'preCST', 'VA', 'CST']:
        train_preliminary.loc[train_preliminary[column].isna(), column] = train_preliminary[column].mean()
    for column in ['preIRF', 'preSRF', 'prePED', 'preHRF', 'continue injection', 'IRF', 'SRF', 'PED', 'HRF']:
        train_preliminary.loc[train_preliminary[column].isna(), column] = 0

    train_preliminary.loc[train_preliminary['continue injection'] > 1.0, 'continue injection'] = 1.0
    train_preliminary.loc[train_preliminary['continue injection'] < 0.0, 'continue injection'] = 0.0

    train_case = train_preliminary[
        ['patient ID', 'gender', 'age', 'diagnosis', 'anti-VEGF', 'continue injection']].copy()

    train_pre = train_preliminary[
        ['patient ID', 'preVA', 'VA', 'preCST', 'CST', 'preIRF', 'preSRF', 'prePED', 'preHRF']].copy()
    train_pre.rename(
        columns={'preIRF': 'IRF', 'preSRF': 'SRF', 'prePED': 'PED', 'preHRF': 'HRF'},
        inplace=True)
    train_pre = train_pre.merge(train_case, on='patient ID', sort=True)
    train_pre = train_pre_image_data.merge(train_pre, on='patient ID', sort=True)

    train_post = train_preliminary[['patient ID', 'preVA', 'VA', 'preCST', 'CST', 'IRF', 'SRF', 'PED', 'HRF']].copy()
    train_post = train_post.merge(train_case, on='patient ID', sort=True)
    train_post = train_post_image_data.merge(train_post, on='patient ID', sort=True)

    train_preliminary = pd.concat([train_pre, train_post], sort=True)
    train_preliminary = train_preliminary[['patient ID', 'gender', 'age', 'diagnosis', 'anti-VEGF', 'preVA', 'VA',
                                           'preCST', 'CST', 'IRF', 'SRF', 'PED', 'HRF', 'continue injection', 'L0R1',
                                           'injection', 'image name', 'after', 'final', 'data_type', 'processed_path',
                                           'source_path']]
    return train_preliminary


def _processing_data_source_final_test():
    """
    处理复赛数据
    :return:
    """
    print('_processing_data_source_final_test')
    image_data = pd.read_csv(config.PROCESSED_IMAGE_CSV_PATH)
    test_image_data = image_data.loc[(image_data['data_type'] == 'test') & (image_data['final'] == 1)].copy()

    test_case = pd.read_csv(config.SOURCE_TEST_CSV_PATH_FINAL)

    test_case.loc[test_case['gender'] == 'Male', 'gender'] = 1
    test_case.loc[test_case['gender'] == 'Male ', 'gender'] = 1
    test_case.loc[test_case['gender'] == 'Female', 'gender'] = 2

    test_case.loc[test_case['diagnosis'] == 'CNVM', 'diagnosis'] = 1
    test_case.loc[test_case['diagnosis'] == 'PCV', 'diagnosis'] = 2
    test_case.loc[test_case['diagnosis'] == 'DME', 'diagnosis'] = 3
    test_case.loc[test_case['diagnosis'] == 'RVO', 'diagnosis'] = 4
    test_case.loc[test_case['diagnosis'] == 'CME', 'diagnosis'] = 5

    test_case.loc[test_case['anti-VEGF'] == 'Avastin', 'anti-VEGF'] = 1
    test_case.loc[test_case['anti-VEGF'] == 'Razumab', 'anti-VEGF'] = 2
    test_case.loc[test_case['anti-VEGF'] == 'Accentrix', 'anti-VEGF'] = 2
    test_case.loc[test_case['anti-VEGF'] == 'Eylea', 'anti-VEGF'] = 3
    test_case.loc[test_case['anti-VEGF'] == 'Tricort', 'anti-VEGF'] = 5
    test_case.loc[test_case['anti-VEGF'] == 'Ozrudex', 'anti-VEGF'] = 10
    test_case.loc[test_case['anti-VEGF'] == 'Ozurdex', 'anti-VEGF'] = 10
    test_case.loc[test_case['anti-VEGF'] == 'Pagenax', 'anti-VEGF'] = 0

    test_final = test_image_data.merge(test_case, on='patient ID', sort=True)

    test = test_final[['patient ID', 'gender', 'age', 'diagnosis', 'anti-VEGF', 'preVA', 'L0R1', 'injection',
                       'image name', 'after', 'final', 'data_type', 'processed_path', 'source_path']]

    return test


def _processing_data_source_preliminary_test():
    """
    处理初赛数据
    :return:
    """
    print('_processing_data_source_preliminary_test')
    image_data = pd.read_csv(config.PROCESSED_IMAGE_CSV_PATH)
    test_image_data = image_data.loc[(image_data['data_type'] == 'test') & (image_data['final'] == 0)].copy()
    test_preliminary = pd.read_csv(config.SOURCE_TEST_CSV_PATH_PRELIMINARY)

    for column in ['gender', 'age', 'diagnosis', 'preVA', 'anti-VEGF']:
        test_preliminary.loc[test_preliminary[column].isna(), column] = test_preliminary[column].mean()

    test_preliminary = test_image_data.merge(test_preliminary, on='patient ID', sort=True)

    test = test_preliminary[
        ['patient ID', 'gender', 'age', 'diagnosis', 'anti-VEGF', 'preVA', 'L0R1', 'injection', 'image name', 'after',
         'final', 'data_type', 'processed_path', 'source_path']]

    return test


def processing_data():
    crop_img()

    test_preliminary = _processing_data_source_preliminary_test()
    test_final = _processing_data_source_final_test()
    test = pd.concat([test_preliminary, test_final], sort=True)
    test = test[['patient ID', 'gender', 'age', 'diagnosis', 'anti-VEGF', 'preVA', 'L0R1', 'injection',
                 'image name', 'after', 'final', 'data_type', 'processed_path', 'source_path']]
    test.to_csv(config.PROCESSED_TEST_CSV_PATH, index=False)

    train_preliminary = _processing_data_source_preliminary_train()
    train_final = _processing_data_source_final_train()
    train = pd.concat([train_preliminary, train_final], sort=True)
    train = train[['patient ID', 'gender', 'age', 'diagnosis', 'anti-VEGF', 'preVA', 'VA', 'preCST', 'CST', 'IRF',
                   'SRF', 'PED', 'HRF', 'continue injection', 'L0R1', 'injection', 'image name', 'after', 'final',
                   'data_type', 'processed_path', 'source_path']]
    train.to_csv(config.PROCESSED_TRAIN_CSV_PATH, index=False)


def plot_log(title, xlabel, ylabel, data1, label1, data2=None, label2=None, log_path=None):
    """
    绘图
    :param title:
    :param xlabel:
    :param ylabel:
    :param data1:
    :param label1:
    :param data2:
    :param label2:
    :param log_path:
    :return:
    """
    if not os.path.exists(log_path):
        return
    x = [i for i in range(len(data1))]
    plt.plot(x, data1, c='r', lw=1, label=label1)
    if data2:
        plt.plot(x, data2, c='b', lw=1, label=label2)
        plt.legend(prop={'size': 10})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(os.path.join(log_path, f'{title}.png'))
    plt.close()
