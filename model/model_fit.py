import os
import numpy as np
import pandas as pd

from config import MODEL_RESULT_SAVE_PATH
from data_load import dataLoad

from model_base import BaseModel
from model_LeNet import LeNetModel
from model_VGGNet import VGGNet11


def model_fit_cst(model: BaseModel, epochs=10, batch_size=64, read_img=False, to_float=False):
    train_data = dataLoad.get_train_data_cst_all(read_img=read_img)
    X = np.array(train_data['feature'].tolist())
    print('train:', X.shape)
    Y = train_data['label']
    # for i in range(10):
    if read_img:
        new_model_name = model.fit_img_array(X, Y, epochs=epochs, batch_size=batch_size)
        model_name = new_model_name
        predict_y = model.predict_img_array(X)
    else:
        new_model_name = model.fit_img_path(X, Y, epochs=epochs, batch_size=batch_size, to_float=to_float)
        model_name = new_model_name
        predict_y = model.predict_img_path(X, to_float=to_float)

    predict_data = pd.concat([train_data.drop('feature', axis=1), predict_y], axis=1)
    predict_data.to_csv(os.path.join(MODEL_RESULT_SAVE_PATH, f'train-{model_name}.csv'), index=False)
    print(model_name)
    return model_name


if __name__ == '__main__':
    model_name = 'LeNet-09252222-e(50)-b(64)-eta(1e-05)-loss(4.2496).h5'
    model_name = model_fit_cst(LeNetModel(learning_rate=1e-6, load_model_name=model_name), epochs=500)
    #
    # model_fit_cst(VGGNet11(learning_rate=1e-3, load_model_name=model_name), epochs=50, batch_size=16)
