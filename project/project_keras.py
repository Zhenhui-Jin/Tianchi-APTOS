import os

import numpy as np
import pandas as pd

from data_load import dataLoad
from model.model_keras import CSTModel

from config import KERAS_PREDICT_SAVE_PATH


class Project:
    def __init__(self, cst_model_name):
        self.cst_model_name = cst_model_name

    def predict_pre_cst(self):
        test_data = dataLoad.get_test_data_cst_all(read_img=True)
        X = np.array(test_data['feature'].tolist())

        model = CSTModel(load_model_name=self.cst_model_name, trainable=False)
        predict_y = model.predict_img_array(X)
        predict_data = pd.concat([test_data.drop('feature', axis=1), predict_y], axis=1)

        pre_cst_data = predict_data[predict_data['type'] == 'preCST'].copy()
        pre_cst_data.drop(['type'], axis=1, inplace=True)
        pre_cst_data.rename({'predict': 'preCST'}, axis=1, inplace=True)
        pre_cst_data.set_index('patient ID', inplace=True)

        cst_data = predict_data[predict_data['type'] == 'CST'].copy()
        cst_data.drop(['type'], axis=1, inplace=True)
        cst_data.rename({'predict': 'CST'}, axis=1, inplace=True)
        cst_data.set_index('patient ID', inplace=True)

        predict_data = pd.concat([pre_cst_data, cst_data], axis=1)

        return predict_data


if __name__ == '__main__':
    cst_model_name = 'CSTModel-09212209-e(100)-b(64)-eta(1e-06)-loss(480.9583).h5'
    project = Project(cst_model_name=cst_model_name)

    # patient ID,preCST,VA,continue injection,CST,IRF,SRF,HRF

    predict_data = project.predict_pre_cst()
    predict_data.loc[predict_data['preCST'].isna(), 'preCST'] = 0

    predict_data['VA'] = 0
    predict_data['continue injection'] = 0
    predict_data['IRF'] = 0
    predict_data['SRF'] = 0
    predict_data['HRF'] = 0

    predict_data.reset_index(inplace=True)
    predict_data.rename({'index': 'patient ID'}, axis=1, inplace=True)

    predict_data = predict_data[['patient ID', 'preCST', 'VA', 'continue injection', 'CST', 'IRF', 'SRF', 'HRF']]
    predict_data.to_csv(os.path.join(KERAS_PREDICT_SAVE_PATH, f'{cst_model_name}.csv'), index=False)
