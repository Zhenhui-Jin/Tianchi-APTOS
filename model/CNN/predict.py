import os
import time

import pandas as pd

import config
from model.CNN.aptos_model import APTOSModel, CSVModel


def eval_after():
    TEST_DATA_AFTER_PATH = os.path.join(config.DATA_PATH, 'test_data_after.csv')

    SAVE_MODEL_PATH = os.path.join(config.MODEL_SAVE_PATH, 'CNN', f'{time.strftime("%Y%m%d")}')
    LOAD_MODEL_PATH = os.path.join(SAVE_MODEL_PATH, 'AfterModel.pt')

    model = APTOSModel('AfterModel', 'ImgPath', 'CST',
                       ['IRF', 'SRF', 'PED', 'HRF'],
                       save_model_path=SAVE_MODEL_PATH)

    data = model.eval(index_column='patient ID', data_csv_path=TEST_DATA_AFTER_PATH, load_model_path=LOAD_MODEL_PATH)
    return data


def eval_before():
    TEST_DATA_BEFORE_PATH = os.path.join(config.DATA_PATH, 'test_data_before.csv')

    SAVE_MODEL_PATH = os.path.join(config.MODEL_SAVE_PATH, 'CNN', f'{time.strftime("%Y%m%d")}')
    LOAD_MODEL_PATH = os.path.join(SAVE_MODEL_PATH, 'BeforeModel.pt')

    model = APTOSModel('BeforeModel', 'ImgPath', 'preCST',
                       ['preIRF', 'preSRF', 'prePED', 'preHRF'],
                       save_model_path=SAVE_MODEL_PATH)

    data = model.eval(index_column='patient ID', data_csv_path=TEST_DATA_BEFORE_PATH, load_model_path=LOAD_MODEL_PATH)
    return data


def eval_csv(test_data_path):
    feature_columns = ['gender', 'age', 'diagnosis', 'preVA', 'anti-VEGF', 'preCST',
                       'preIRF', 'preSRF', 'prePED', 'preHRF', 'CST', 'IRF', 'SRF', 'PED', 'HRF']
    label_regression = 'VA'
    label_classify = ['continue injection']

    SAVE_MODEL_PATH = os.path.join(config.MODEL_SAVE_PATH, 'CNN', f'{time.strftime("%Y%m%d")}')
    LOAD_MODEL_PATH = os.path.join(SAVE_MODEL_PATH, 'CSVModel.pt')

    model = CSVModel(model_name='CSVModel', index_column='patient ID',
                     feature_columns=feature_columns,
                     label_regression=label_regression,
                     label_classify=label_classify, save_model_path=SAVE_MODEL_PATH)

    data = model.eval(data_csv_path=test_data_path, load_model_path=LOAD_MODEL_PATH)
    return data


if __name__ == '__main__':
    TEST_DATA_PATH = os.path.join(config.TEST_DATA_FILE_NEW, 'PreliminaryValidationSet_Info.csv')
    test_data = pd.read_csv(TEST_DATA_PATH)
    # test_data_path = os.path.join(config.MODEL_SAVE_PATH, 'CNN', f'predict.csv')
    test_data_path = os.path.join(config.MODEL_SAVE_PATH, 'CNN', f'{time.strftime("%Y%m%d")}',
                                  f'predict-{time.strftime("%H%M")}.csv')
    submit_data_path = os.path.join(config.MODEL_SAVE_PATH, 'CNN', f'{time.strftime("%Y%m%d")}',
                                    f'predict-{time.strftime("%H%M")}-submit.csv')

    test_data.to_csv(test_data_path, index=False)
    print(test_data.info())
    print(test_data)

    after = eval_after()
    test_data = pd.merge(test_data, after, on='patient ID')
    test_data.to_csv(test_data_path, index=False)

    before = eval_before()
    test_data = pd.merge(test_data, before, on='patient ID')
    test_data.to_csv(test_data_path, index=False)

    test = eval_csv(test_data_path)
    test_data = pd.merge(test_data, test, on='patient ID')
    test_data.to_csv(test_data_path, index=False)

    submit_data = test_data[['patient ID', 'preCST', 'VA', 'continue injection', 'CST', 'IRF', 'SRF', 'HRF']]
    submit_data.to_csv(submit_data_path, index=False)
