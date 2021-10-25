import os
import numpy as np
import pandas as pd

from config import MODEL_RESULT_SAVE_PATH, CST_TRAIN_DATA_FILE, CST_TEST_DATA_FILE
from data_load import DataLoad

from model_base import BaseModel
from model_LeNet import LeNetModel
from model_VGGNet import VGGNet11


def model_fit_cst(model: BaseModel, epochs=10, batch_size=64, read_img=False, to_float=False):
    train_data = model.dataLoad.get_train_data_cst_all(read_img=read_img)
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
    dataLoad = DataLoad(CST_TRAIN_DATA_FILE, CST_TEST_DATA_FILE)
    # model_name = 'LeNet-09252354-e(50)-b(64)-eta(1e-06)-loss(59.7677).h5'
    # model_name = model_fit_cst(LeNetModel(learning_rate=1e-6, load_model_name=model_name), epochs=500)
    #
    model_name = 'CST-VGGNet11-10050604-e(50)-b(16)-eta(1e-07)-loss(0.2738).h5'
    for eta in [1e-8, 1e-8]:
        model = VGGNet11(model_name='CST-VGGNet11', learning_rate=eta, load_model_name=model_name, dataLoad=dataLoad)
        model_name = model_fit_cst(model, epochs=50, batch_size=16)
