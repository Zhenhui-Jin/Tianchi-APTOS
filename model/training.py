import pandas as pd

import config
from model.model import ImageModel, CSVModel
from model_config import ImageConfig, CSVConfig


def _get_train_data(data):
    train = data.loc[data['data_type'] == 'train'].copy()
    return train


def train_image(**kwargs):
    train_data = pd.read_csv(config.PROCESSED_TRAIN_CSV_PATH)
    train = _get_train_data(train_data)

    print('train_image')

    train_before = train.loc[train['after'] == 0].copy()
    train_before.drop('CST', axis=1, inplace=True)
    train_before.rename(columns={'preCST': 'CST'}, inplace=True)

    train_after = train.loc[train['after'] == 1].copy()
    train_after.drop('preCST', axis=1, inplace=True)

    train = pd.concat([train_before, train_after])

    image_config = ImageConfig(**kwargs)
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
