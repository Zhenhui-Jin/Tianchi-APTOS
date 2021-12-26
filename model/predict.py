import os
import time

import pandas as pd

import config
from model.model import ImageModel, CSVModel
from model_config import ImageConfig, CSVConfig


def _predict_image(model_load_path, **kwargs):
    print('predict_image')
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


def predict(model_image_path, model_csv_path):
    result_path = os.path.join(config.PREDICT_RESULT_PATH, f'{time.strftime("%Y%m%d%H%M")}')
    os.makedirs(result_path, exist_ok=True)

    result_stage1 = os.path.join(result_path, 'submit_stage1.csv')
    result_stage2_case = os.path.join(result_path, 'submit_stage2_case.csv')
    result_stage2_pic = os.path.join(result_path, 'submit_stage2_pic.csv')

    stage2_pic, predict_data, = _predict_image(model_image_path)
    stage2_pic.to_csv(result_stage2_pic, index=False)

    predict_data.to_csv('predict_image.csv')
    predict_data = predict_csv(predict_data, model_csv_path)
    predict_data.to_csv('predict_csv.csv', index=False)
