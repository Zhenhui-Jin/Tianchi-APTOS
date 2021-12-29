import pandas as pd

import config
from model.model import ImageModel, CSVModel
from model_config import ImageConfig, CSVConfig


def _get_train_data(data):
    train = data.loc[data['data_type'] == 'train'].copy()
    return train


def train_image(after: int = None, **kwargs):
    train_data = pd.read_csv(config.PROCESSED_TRAIN_CSV_PATH)
    train = _get_train_data(train_data)

    print('train_image')
    if after is not None:
        train = train.loc[train['after'] == after].copy()
        if after == 1:
            train.drop('preCST', axis=1, inplace=True)
            name = 'Model-Image-After'
        else:
            train.drop('CST', axis=1, inplace=True)
            train.rename(columns={'preCST': 'CST'}, inplace=True)
            name = 'Model-Image-Before'
    else:
        name = 'Model-Image-All'

    image_config = ImageConfig(name=name, **kwargs)
    image_model = ImageModel(image_config)
    image_model.train(train)

    return image_config.model_save_path


def train_csv(**kwargs):
    print('train_csv')
    train_data = pd.read_csv(config.PROCESSED_TRAIN_CSV_PATH)
    train = _get_train_data(train_data)

    data = train[['patient ID', 'gender', 'age', 'diagnosis', 'anti-VEGF', 'continue injection', 'L0R1']]
    data = data.groupby(['patient ID']).mean()

    train_before = train.loc[train['after'] == 0][['patient ID', 'preVA', 'preCST', 'IRF', 'SRF', 'PED', 'HRF']]

    train_before: pd.DataFrame = train_before.groupby(['patient ID']).mean()
    train_before.rename(columns={'IRF': 'preIRF', 'SRF': 'preSRF', 'PED': 'prePED', 'HRF': 'preHRF'}, inplace=True)
    data = data.merge(train_before, on=['patient ID'])

    train_after = train.loc[train['after'] == 1][['patient ID', 'VA', 'CST', 'IRF', 'SRF', 'PED', 'HRF']]
    train_after = train_after.groupby(['patient ID']).mean()
    train = data.merge(train_after, on='patient ID')

    csv_config = CSVConfig(**kwargs)
    csv_model = CSVModel(csv_config)
    csv_model.train(train)

    return csv_config.model_save_path
