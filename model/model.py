import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset import ImageDataset
from model.loss import ImageLoss
from model.module import ImageModule
from model_config import ImageConfig
from processing import data_factory


class ImageModel:
    def __init__(self, config: ImageConfig):
        self.config = config

    def train(self, train_data):
        self.config.training = True
        device = torch.device(self.config.device)

        data_loader = DataLoader(ImageDataset(self.config, train_data), batch_size=self.config.batch_size, shuffle=True,
                                 num_workers=5)
        model = ImageModule()
        model = model.to(device)
        model.train()
        if self.config.model_load_path and os.path.exists(self.config.model_load_path):
            print('load model')
            model.load_state_dict(torch.load(self.config.model_load_path))

        # print(model)
        lossFn = ImageLoss()
        optim = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        loss_epochs = []
        for epoch in range(self.config.epochs):
            loss_batchs = []
            for images, labels in tqdm(data_loader):
                optim.zero_grad()
                images = images.to(device)
                labels[0] = labels[0].to(device)
                labels[1] = labels[1].to(device)
                y = model(images)
                loss = lossFn(y, labels)
                loss.backward()
                optim.step()
                with torch.no_grad():
                    loss_batch = loss.detach().cpu().numpy()
                    loss_batchs.append(loss_batch)

                # if len(loss_batchs) % 100 == 0:
                #     torch.save(model.state_dict(), self.config.model_save_path)
            loss_epochs.append(np.mean(loss_batchs))
            print(f'{epoch + 1}/{self.config.epochs}\tloss:{np.mean(loss_batchs)}')

            torch.save(model.state_dict(), self.config.model_save_path)

        log_name = f'{self.config.model_name}-loss-epoch({self.config.epochs})-eta({self.config.learning_rate})-{time.strftime("%Y%m%d%H%M")}'
        data_factory.plot_log(log_name, 'epoch', 'loss', loss_epochs, 'train_loss', log_path=self.config.model_log_path)

        return loss_epochs

    def eval(self, data: pd.DataFrame) -> pd.DataFrame:
        self.config.training = False
        device = torch.device(self.config.device)

        data_loader = DataLoader(ImageDataset(self.config, data), batch_size=self.config.batch_size, shuffle=False,
                                 num_workers=5)

        model = ImageModule()
        model = model.to(device)
        model.eval()

        if self.config.model_load_path and os.path.exists(self.config.model_load_path):
            model.load_state_dict(torch.load(self.config.model_load_path))

        eval_data = pd.DataFrame(columns=[self.config.label_regression, *self.config.label_classify])
        for images in tqdm(data_loader):
            images = images.to(device)

            y_regression, y_classify = model(images)

            y_regression = y_regression * 100.0

            y_regression = pd.DataFrame(y_regression.cpu().detach().numpy(), columns=[self.config.label_regression])
            y_classify = pd.DataFrame(y_classify.cpu().detach().numpy(), columns=[*self.config.label_classify])

            batch_data = pd.concat([y_regression, y_classify], axis=1)

            eval_data = pd.concat([eval_data, batch_data])

        eval_data.set_index(data[self.config.index_column], inplace=True)
        eval_data = data.merge(eval_data, on=self.config.index_column, sort=True)
        return eval_data


class CSVModule(nn.Module):
    def __init__(self, in_features: int, classify_sum: int = 2):
        super(CSVModule, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
        )

        self.regression = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

        self.classify = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, classify_sum),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.layers(x)
        y_regression = self.regression(x)
        y_classify = self.classify(x)
        return y_regression, y_classify
