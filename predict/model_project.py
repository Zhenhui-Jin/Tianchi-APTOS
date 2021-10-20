import os

import numpy as np
import pandas as pd

from config import PREDICT_SAVE_PATH, CST_TRAIN_DATA_FILE, CST_TEST_DATA_FILE
from data_load import DataLoad
from model.model_VGGNet import VGGNet11


class Predict:

    def predict_pre_cst(self, model):
        test_data = model.dataLoad.get_test_data_cst_all(read_img=False)
        X = np.array(test_data['feature'].tolist())

        predict_y = model.predict_img_path(X)
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
    dataLoad = DataLoad(CST_TRAIN_DATA_FILE, CST_TEST_DATA_FILE)
    # model = CSTModel(load_model_name=self.cst_model_name, trainable=False)
    # net_model = LeNetModel(load_model_name='VGGNet11-09281646-e(100)-b(16)-eta(1e-05)-loss(6.3234).h5', trainable=False)
    net_model = VGGNet11(load_model_name='CST-VGGNet11-10050604-e(50)-b(16)-eta(1e-07)-loss(0.2738).h5',
                         trainable=False, dataLoad=dataLoad)

    predict = Predict()

    # patient ID,preCST,VA,continue injection,CST,IRF,SRF,HRF

    predict_data = predict.predict_pre_cst(net_model)
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
