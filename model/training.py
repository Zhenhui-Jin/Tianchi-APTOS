import os
import pandas as pd
import config
from model.model import ImageModel
from model_config import ImageConfig


def train_image(after: int = -1, final: int = -1, **kwargs):
    train_data = pd.read_csv(config.PROCESSED_TRAIN_CSV_PATH)
    name = 'Model'
    train = train_data.loc[train_data['data_type'] == 'train'].copy()

    if after == 0 or after == 1:
        train = train_data.loc[train_data['after'] == after].copy()
        if after == 1:
            name += 'After'
        else:
            name += 'Before'
    else:
        name += 'All'

    if final == 0 or final == 1:
        train = train.loc[train['final'] == final].copy()
        if final == 1:
            name += '-final'
        else:
            name += '-preliminary'
    else:
        name += '-all'

    print('train', name)

    image_config = ImageConfig(name=name, **kwargs)
    image_model = ImageModel(image_config)
    image_model.train(train)

    return image_config.model_save_path


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
