import os

import numpy as np
import pandas as pd

from config import KERAS_MODEL_RESULT_SAVE_PATH
from data_load import dataLoad

from model_base import BaseModel
from model_LeNet import LeNetModel


def model_fit_cst(model: BaseModel, epochs=10, read_img=False):
    train_data = dataLoad.get_train_data_cst_all(read_img=read_img)
    X = np.array(train_data['feature'].tolist())
    print('train:', X.shape)
    Y = train_data['label']
    # for i in range(10):
    if read_img:
        new_model_name = model.fit_img_array(X, Y, epochs=epochs)
        model_name = new_model_name
        predict_y = model.predict_img_array(X)
    else:
        new_model_name = model.fit_img_path(X, Y, epochs=epochs, to_float=True)
        model_name = new_model_name
        predict_y = model.predict_img_path(X, to_float=True)

    predict_data = pd.concat([train_data.drop('feature', axis=1), predict_y], axis=1)
    predict_data.to_csv(os.path.join(KERAS_MODEL_RESULT_SAVE_PATH, f'train-{model_name}.csv'), index=False)
    print(model_name)


if __name__ == '__main__':
    model_name = 'LeNet-09232034-e(50)-b(64)-eta(0.0001)-loss(41458.1824).h5'
    model_fit_cst(LeNetModel(learning_rate=1e-4, load_model_name=model_name), epochs=50)
