import os
import time

import pandas as pd

import config
from model.model import ImageModel, CSVModel
from model_config import ImageConfig, CSVConfig


def _predict_image_before_after(before_model_load_path, after_model_load_path, **kwargs):
    print('predict_image_before_after')
    test_data = pd.read_csv(config.PROCESSED_TEST_CSV_PATH)
    test_before: pd.DataFrame = test_data.loc[test_data['after'] == 0].copy()
    test_after: pd.DataFrame = test_data.loc[test_data['after'] == 1].copy()
    test_before.reset_index(drop=True, inplace=True)
    test_after.reset_index(drop=True, inplace=True)

    before_config = ImageConfig(name='Model-Image-Before', training=False, model_load_path=before_model_load_path,
                                **kwargs)
    before_model = ImageModel(before_config)
    predict_before = before_model.eval(test_before)

    after_config = ImageConfig(name='Model-Image-After', training=False, model_load_path=after_model_load_path,
                               **kwargs)
    after_model = ImageModel(after_config)
    predict_after = after_model.eval(test_after)

    predict_data = pd.concat([predict_after, predict_before])

    predict_final = predict_data.loc[predict_data['final'] == 1].copy()

    submit_stage2_pic = pd.read_csv('data/submit/submit_stage2_pic.csv')
    submit_stage2_pic.drop(['IRF', 'SRF', 'PED'], axis=1, inplace=True)
    submit_stage2_predict = predict_final[['patient ID', 'injection', 'image name', 'IRF', 'SRF', 'PED', 'HRF']]
    submit_stage2_pic = submit_stage2_pic.merge(submit_stage2_predict, on=['patient ID', 'injection', 'image name'])

    predict_before = predict_data.loc[predict_data['after'] == 0][['patient ID', 'CST', 'IRF', 'SRF', 'PED', 'HRF']]
    predict_before = predict_before.groupby(['patient ID']).mean()
    predict_before.rename(columns={'CST': 'preCST', 'IRF': 'preIRF', 'SRF': 'preSRF', 'PED': 'prePED', 'HRF': 'preHRF'},
                          inplace=True)

    predict_after = predict_data.loc[predict_data['after'] == 1][['patient ID', 'CST', 'IRF', 'SRF', 'PED', 'HRF']]
    predict_after = predict_after.groupby(['patient ID']).mean()
    predict_after = predict_before.merge(predict_after, on=['patient ID'])

    predict_data = predict_data[['patient ID', 'gender', 'age', 'diagnosis', 'anti-VEGF', 'preVA', 'L0R1']]
    predict_data = predict_data.groupby(['patient ID']).mean()
    predict_data = pd.concat([predict_data, predict_after], axis=1)

    return submit_stage2_pic, predict_data


def _predict_image_all(model_load_path, **kwargs):
    print('predict_image_all')
    test_data = pd.read_csv(config.PROCESSED_TEST_CSV_PATH)

    image_config = ImageConfig(training=False, model_load_path=model_load_path, **kwargs)
    image_model = ImageModel(image_config)
    predict_data = image_model.eval(test_data)

    predict_final = predict_data.loc[predict_data['final'] == 1].copy()

    submit_stage2_pic = pd.read_csv('data/submit/submit_stage2_pic.csv')
    submit_stage2_pic.drop(['IRF', 'SRF', 'PED'], axis=1, inplace=True)
    submit_stage2_predict = predict_final[['patient ID', 'injection', 'image name', 'IRF', 'SRF', 'PED', 'HRF']]
    submit_stage2_pic = submit_stage2_pic.merge(submit_stage2_predict, on=['patient ID', 'injection', 'image name'])

    predict_before = predict_data.loc[predict_data['after'] == 0][['patient ID', 'CST', 'IRF', 'SRF', 'PED', 'HRF']]
    predict_before = predict_before.groupby(['patient ID']).mean()
    predict_before.rename(columns={'CST': 'preCST', 'IRF': 'preIRF', 'SRF': 'preSRF', 'PED': 'prePED', 'HRF': 'preHRF'},
                          inplace=True)

    predict_after = predict_data.loc[predict_data['after'] == 1][['patient ID', 'CST', 'IRF', 'SRF', 'PED', 'HRF']]
    predict_after = predict_after.groupby(['patient ID']).mean()
    predict_after = predict_before.merge(predict_after, on=['patient ID'])

    predict_data = predict_data[['patient ID', 'gender', 'age', 'diagnosis', 'anti-VEGF', 'preVA', 'L0R1']]
    predict_data = predict_data.groupby(['patient ID']).mean()
    predict_data = pd.concat([predict_data, predict_after], axis=1)

    return submit_stage2_pic, predict_data


def predict_csv(test_data, model_load_path, **kwargs):
    print('predict_csv')
    csv_config = CSVConfig(model_load_path=model_load_path, training=False, **kwargs)
    csv_model = CSVModel(csv_config)
    predict_data = csv_model.eval(test_data)
    return predict_data


def predict_all(model_image_path, model_csv_path):
    result_path = os.path.join(config.PREDICT_RESULT_PATH, f'{time.strftime("%Y%m%d%H%M%D")}')
    os.makedirs(result_path, exist_ok=True)

    result_stage1 = os.path.join(result_path, 'submit_stage1.csv')
    result_stage2_case = os.path.join(result_path, 'submit_stage2_case.csv')
    result_stage2_pic = os.path.join(result_path, 'submit_stage2_pic.csv')

    stage2_pic, predict_data, = _predict_image_all(model_image_path)
    stage2_pic.to_csv(result_stage2_pic, index=False)

    predict_data = predict_csv(predict_data, model_csv_path)
    predict_data['VA'] = abs(predict_data['VA'])
    submit_stage1 = pd.read_csv('data/submit/submit_stage1.csv')
    submit_stage1 = submit_stage1[['patient ID']]
    submit_stage1 = submit_stage1.merge(predict_data, on='patient ID')
    submit_stage1 = submit_stage1[['patient ID', 'preCST', 'VA', 'continue injection', 'CST', 'IRF', 'SRF', 'HRF']]
    submit_stage1.to_csv(result_stage1, index=False)

    stage2_case = pd.read_csv('data/submit/submit_stage2_case.csv')
    stage2_case = stage2_case[['patient ID']]
    stage2_case = stage2_case.merge(predict_data, on='patient ID')
    stage2_case = stage2_case[['patient ID', 'VA', 'continue injection', 'preCST', 'CST']]
    stage2_case.to_csv(result_stage2_case, index=False)


def predict_before_after(before_model_load_path, after_model_load_path, model_csv_path):
    result_path = os.path.join(config.PREDICT_RESULT_PATH, f'{time.strftime("%Y%m%d%H%M")}')

    result_stage1 = os.path.join(result_path, 'submit_stage1.csv')
    result_stage2_case = os.path.join(result_path, 'submit_stage2_case.csv')
    result_stage2_pic = os.path.join(result_path, 'submit_stage2_pic.csv')

    stage2_pic, predict_data, = _predict_image_before_after(before_model_load_path, after_model_load_path)

    predict_data = predict_csv(predict_data, model_csv_path)

    os.makedirs(result_path, exist_ok=True)
    stage2_pic.to_csv(result_stage2_pic, index=False)
    predict_data['VA'] = abs(predict_data['VA'])
    submit_stage1 = pd.read_csv('data/submit/submit_stage1.csv')
    submit_stage1 = submit_stage1[['patient ID']]
    submit_stage1 = submit_stage1.merge(predict_data, on='patient ID')
    submit_stage1 = submit_stage1[['patient ID', 'preCST', 'VA', 'continue injection', 'CST', 'IRF', 'SRF', 'HRF']]
    submit_stage1.to_csv(result_stage1, index=False)

    stage2_case = pd.read_csv('data/submit/submit_stage2_case.csv')
    stage2_case = stage2_case[['patient ID']]
    stage2_case = stage2_case.merge(predict_data, on='patient ID')
    stage2_case = stage2_case[['patient ID', 'VA', 'continue injection', 'preCST', 'CST']]
    stage2_case.to_csv(result_stage2_case, index=False)
    return result_path
