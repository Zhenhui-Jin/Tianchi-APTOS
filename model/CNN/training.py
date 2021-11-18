import os

import config
from model.CNN.aptos_model import APTOSModel, CSVModel


def train_after():
    TRAIN_DATA_AFTER_PATH = os.path.join(config.DATA_PATH, 'train_data_after.csv')

    SAVE_MODEL_PATH = os.path.join(config.MODEL_SAVE_PATH, 'CNN')
    LOAD_MODEL_PATH = os.path.join(SAVE_MODEL_PATH, 'AfterModel.pt')

    model = APTOSModel('AfterModel', 'ImgPath', 'preCST',
                       ['preIRF', 'preSRF', 'prePED', 'preHRF'],
                       save_model_path=SAVE_MODEL_PATH)

    for eta in [1e-4, 1e-4, 1e-4, 1e-4, 1e-4]:
        model.train(epochs=100, batch_size=50, learning_rate=eta,
                    device='cuda',
                    data_csv_path=TRAIN_DATA_AFTER_PATH,
                    load_model_path=LOAD_MODEL_PATH)


def train_before():
    TRAIN_DATA_BEFORE_PATH = os.path.join(config.DATA_PATH, 'train_data_before.csv')

    SAVE_MODEL_PATH = os.path.join(config.MODEL_SAVE_PATH, 'CNN')
    LOAD_MODEL_PATH = os.path.join(SAVE_MODEL_PATH, 'BeforeModel.pt')

    model = APTOSModel('BeforeModel', 'ImgPath', 'CST',
                       ['IRF', 'SRF', 'PED', 'HRF'],
                       save_model_path=SAVE_MODEL_PATH)

    for eta in [1e-4, 1e-4, 1e-4, 1e-4, 1e-4]:
        model.train(epochs=100, batch_size=50, learning_rate=eta,
                    device='cuda',
                    data_csv_path=TRAIN_DATA_BEFORE_PATH,
                    load_model_path=LOAD_MODEL_PATH)


def train_csv():
    data_csv_path = os.path.join(config.TRAIN_DATA_FILE_NEW, 'TrainingAnnotation.csv')
    feature_columns = ['gender', 'age', 'diagnosis', 'preVA', 'anti-VEGF', 'preCST',
                       'preIRF', 'preSRF', 'prePED', 'preHRF', 'CST', 'IRF', 'SRF', 'PED', 'HRF']
    label_regression = 'VA'
    label_classify = ['continue injection']

    SAVE_MODEL_PATH = os.path.join(config.MODEL_SAVE_PATH, 'CNN')
    LOAD_MODEL_PATH = os.path.join(SAVE_MODEL_PATH, 'CSVModel.pt')

    model = CSVModel(model_name='CSVModel', index_column='patient ID',
                     feature_columns=feature_columns,
                     label_regression=label_regression,
                     label_classify=label_classify, save_model_path=SAVE_MODEL_PATH)

    for eta in [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]:
        model.train(epochs=100, batch_size=64, learning_rate=eta,
                    device='cuda',
                    data_csv_path=data_csv_path,
                    load_model_path=LOAD_MODEL_PATH)


if __name__ == '__main__':
    train_csv()
    train_before()
    train_after()
