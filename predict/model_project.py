import os

import numpy as np
import pandas as pd

from data_load import dataLoad

from config import PREDICT_SAVE_PATH
from model.model_LeNet import LeNetModel


class Predict:
    def __init__(self, model):
        self.model = model

    def predict_pre_cst(self):
        test_data = dataLoad.get_test_data_cst_all(read_img=True)
        X = np.array(test_data['feature'].tolist())

        predict_y = self.model.predict_img_array(X)
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
    # model = CSTModel(load_model_name=self.cst_model_name, trainable=False)
    net_model = LeNetModel(load_model_name='LeNet-09252222-e(50)-b(64)-eta(1e-05)-loss(4.2496).h5', trainable=False)

    predict = Predict(model=net_model)

    # patient ID,preCST,VA,continue injection,CST,IRF,SRF,HRF

    predict_data = predict.predict_pre_cst()
    predict_data.loc[predict_data['preCST'].isna(), 'preCST'] = 0

    predict_data['VA'] = 0
    predict_data['continue injection'] = 0
    predict_data['IRF'] = 0
    predict_data['SRF'] = 0
    predict_data['HRF'] = 0

    predict_data.reset_index(inplace=True)
    predict_data.rename({'index': 'patient ID'}, axis=1, inplace=True)

    predict_data = predict_data[['patient ID', 'preCST', 'VA', 'continue injection', 'CST', 'IRF', 'SRF', 'HRF']]
    predict_data.to_csv(os.path.join(PREDICT_SAVE_PATH, f'{net_model.load_model_name}.csv'), index=False)
