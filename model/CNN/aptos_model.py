import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import APTOSDataset
from loss import APTOSLoss
from module import APTOSModule


class APTOSModel:
    def __init__(self, model_name: str, img_column: str, label_regression: str, label_classify: list,
                 save_model_path: str):
        self.model_name = model_name
        self.img_column = img_column
        self.label_regression = label_regression
        self.label_classify = label_classify
        self.save_model_path = save_model_path

    def train(self,
              epochs=5,
              batch_size=16,
              learning_rate=1e-3,
              device: str = 'cpu',
              data_csv_path: str = None,
              load_model_path: str = None):
        device = torch.device(device)
        dataset = APTOSDataset(data_csv_path, self.img_column, self.label_regression, self.label_classify)
        dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3)
        model = APTOSModule()
        model.train()
        model.to(device)
        if load_model_path and os.path.exists(load_model_path):
            print('load model')
            model.load_state_dict(torch.load(load_model_path))
        save_path = None
        if self.save_model_path and os.path.exists(self.save_model_path):
            save_path = os.path.join(self.save_model_path, f'{self.model_name}-{time.strftime("%m%d")}.pt')

        print(model)
        lossFn = APTOSLoss()
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_epochs = []
        for epoch in range(epochs):
            tqdm.write(f'{epoch + 1}/{epochs}')
            loss_batchs = []
            batch = 0
            for images, labels in tqdm(dataLoader):
                images = images.to(device)
                labels[0] = labels[0].to(device)
                labels[1] = labels[1].to(device)
                y = model(images)
                loss = lossFn(y, labels)
                loss.backward()
                optim.step()
                optim.zero_grad()
                with torch.no_grad():
                    loss_batch = loss.detach().cpu().numpy()
                    loss_batchs.append(loss_batch)
                batch += 1
                if batch % 100 == 0:
                    # tqdm.write(f'loss: epoch={epoch + 1} - batch={batch} - loss={loss_batch}')
                    if save_path:
                        torch.save(model.state_dict(), save_path)
            loss_mean = np.mean(loss_batchs)
            tqdm.write(f'loss: epoch={epoch + 1} - loss={loss_mean}')
            loss_epochs.append(loss_mean)
            if save_path:
                torch.save(model.state_dict(), save_path)
        return loss_epochs

    def eval(self, data_csv_path, load_model_path):
        dataset = APTOSDataset(data_csv_path, self.img_column, self.label_regression, self.label_classify,
                               training=False)
        dataLoader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=3)
        model = APTOSModule()
        model.eval()
        if load_model_path and os.path.exists(load_model_path):
            model.load_state_dict(torch.load(load_model_path))
        for images in tqdm(dataLoader):
            y_regression, y_classify = model(images)
            print(y_regression*100, y_classify)


if __name__ == '__main__':
    TRAIN_DATA_AFTER_PATH = os.path.join(config.DATA_PATH, 'train_data_after.csv')
    TEST_DATA_AFTER_PATH = os.path.join(config.DATA_PATH, 'test_data_after.csv')

    SAVE_MODEL_PATH = os.path.join(config.MODEL_SAVE_PATH, 'CNN')
    LOAD_MODEL_PATH = os.path.join(SAVE_MODEL_PATH, 'AfterModel-1104.pt')

    model = APTOSModel('AfterModel', 'ImgPath', 'preCST',
                       ['preIRF', 'preSRF', 'prePED', 'preHRF'],
                       save_model_path=SAVE_MODEL_PATH)

    for eta in [1e-5, 1e-5, 1e-5, 1e-5]:
        model.train(epochs=30, batch_size=24, learning_rate=eta,
                    device='cuda',
                    data_csv_path=TRAIN_DATA_AFTER_PATH,
                    load_model_path=LOAD_MODEL_PATH)

    # model.eval(data_csv_path=TEST_DATA_AFTER_PATH, load_model_path=LOAD_MODEL_PATH)
