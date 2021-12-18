import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset import ImageDataset
from model.loss import APTOSLoss
from model.module import APTOSModule
from model_config import ImageConfig
from processing import data_factory


class ImageModel:
    def __init__(self, config: ImageConfig):
        self.config = config

    def train(self, train_data):
        self.config.training = True
        device = torch.device(self.config.device)
        dataset = ImageDataset(self.config, train_data)
        dataLoader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=8)
        model = APTOSModule()
        model.train()
        model.to(device)
        if self.config.model_load_path and os.path.exists(self.config.model_load_path):
            print('load model')
            model.load_state_dict(torch.load(self.config.model_load_path))

        print(model)
        lossFn = APTOSLoss()
        optim = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        loss_epochs = []
        for epoch in range(self.config.epochs):
            loss_batchs = []
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

                if len(loss_batchs) % 100 == 0:
                    torch.save(model.state_dict(), self.config.model_save_path)
            loss_epochs.append(np.mean(loss_batchs))
            print(f'{epoch + 1}/{self.config.epochs}\tloss:{np.mean(loss_batchs)}')

            # log_name = f'{self.config.model_name}-loss-epoch({epoch + 1})-{time.strftime("%Y%m%d%H%M")}'
            # data_factory.plot_log(log_name, 'batch', 'loss', loss_batchs, 'train_loss',
            #                       log_path=self.config.model_log_path)
            torch.save(model.state_dict(), self.config.model_save_path)
        log_name = f'{self.config.model_name}-loss-epoch-{time.strftime("%Y%m%d%H%M")}'
        data_factory.plot_log(log_name, 'epoch', 'loss', loss_epochs, 'train_loss',
                              log_path=self.config.model_log_path)
        return loss_epochs

    def eval(self, data: pd.DataFrame):
        self.config.training = False
        dataset = ImageDataset(self.config, data)
        dataLoader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=8)

        model = APTOSModule()
        model = model.cpu()
        model.eval()

        if self.config.model_load_path and os.path.exists(self.config.model_load_path):
            model.load_state_dict(torch.load(self.config.model_load_path))

        eval_data = pd.DataFrame(columns=[self.label_regression, *self.label_classify])
        for images in tqdm(dataLoader):
            y_regression, y_classify = model(images)

            y_regression = y_regression * 100

            y_classify[y_classify > 0.5] = 1
            y_classify[y_classify <= 0.5] = 0
            y_classify = y_classify.int()

            y_regression = pd.DataFrame(y_regression.detach().numpy(), columns=[self.label_regression])
            y_classify = pd.DataFrame(y_classify.detach().numpy(), columns=[*self.label_classify], dtype=float)

            batch_data = pd.concat([y_regression, y_classify], axis=1)

            eval_data = pd.concat([eval_data, batch_data])

        eval_data = eval_data.set_index(dataset.data[index_column])

        eval_data = eval_data.groupby([index_column]).mean()

        for label in self.label_classify:
            eval_data.loc[eval_data[label] < 0.5, label] = 0
            eval_data.loc[eval_data[label] >= 0.5, label] = 1
            eval_data = eval_data.astype({label: 'int32'})

        return eval_data
