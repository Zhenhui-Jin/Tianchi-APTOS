import os

from model.model_keras import Model
import pandas as pd
from data_load import dataLoad
from config import KERAS_PREDICT_SAVE_PATH


class Project:

    @staticmethod
    def predict_pre_cst(load_model_name):
        model = Model(load_model_name=load_model_name, trainable=False)
        path = os.path.join(KERAS_PREDICT_SAVE_PATH, f'{load_model_name}.csv')
        patient_id, data = dataLoad.get_pre_cst_test_data()
        predict_y = model.predict(data)

        patient_id = pd.DataFrame(patient_id, columns=['patient ID'])
        predict_y = pd.DataFrame(predict_y, columns=['preCST'])

        predict_data = pd.concat([patient_id, predict_y], axis=1)
        predict_data.to_csv(path, index=False)

        return predict_data


if __name__ == '__main__':
    model_name = 'model-09201112-e100-b64-eta0.0001.h5'
    project = Project()
    project.predict_pre_cst(model_name)
