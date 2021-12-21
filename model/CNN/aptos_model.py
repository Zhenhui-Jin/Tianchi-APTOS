import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import APTOSDataset, CSVDataset
from model.loss import ImageLoss, CSVLoss
from model.module import ImageModule, CSVModule


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
        dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        model = ImageModule()
        model.train()
        model.to(device)
        if load_model_path and os.path.exists(load_model_path):
            print('load model')
            model.load_state_dict(torch.load(load_model_path))
        save_path = None
        if self.save_model_path and os.path.exists(self.save_model_path):
            # save_path = os.path.join(self.save_model_path, f'{self.model_name}-{time.strftime("%m%d")}.pt')
            save_path = os.path.join(self.save_model_path, f'{self.model_name}.pt')

        print(model)
        lossFn = ImageLoss()
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_epochs = []
        for epoch in range(epochs):
            loss_batchs = []
            # batch = 0
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
                # batch += 1
                # if batch % 100 == 0:
                # tqdm.write(f'loss: epoch={epoch + 1} - batch={batch} - loss={loss_batch}')
                # if save_path:
                #     torch.save(model.state_dict(), save_path)
            loss_mean = np.mean(loss_batchs)
            print(f'{self.model_name} {epoch + 1}/{epochs} loss: {loss_mean} eta：{learning_rate}\n')
            loss_epochs.append(loss_mean)
            if save_path:
                torch.save(model.state_dict(), save_path)
        return loss_epochs

    def eval(self, index_column, data_csv_path, load_model_path):
        dataset = APTOSDataset(data_csv_path, self.img_column, self.label_regression, self.label_classify,
                               training=False)
        dataLoader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

        model = ImageModule()
        model = model.cpu()
        model.eval()

        if load_model_path and os.path.exists(load_model_path):
            model.load_state_dict(torch.load(load_model_path))

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


class CSVModel:
    def __init__(self, model_name: str, index_column: str, feature_columns: list, label_regression: str,
                 label_classify: list,
                 save_model_path: str):
        self.model_name = model_name
        self.index_column = index_column
        self.feature_columns = feature_columns
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

        dataset = CSVDataset(csv_path=data_csv_path, index_column=self.index_column,
                             feature_columns=self.feature_columns, label_regression=self.label_regression,
                             label_classify=self.label_classify, training=True)
        dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10, prefetch_factor=20)

        model = CSVModule(in_features=len(self.feature_columns), classify_sum=2)
        model.train()
        model.to(device)
        if load_model_path and os.path.exists(load_model_path):
            print('load model')
            model.load_state_dict(torch.load(load_model_path))
        save_path = None
        if self.save_model_path and os.path.exists(self.save_model_path):
            save_path = os.path.join(self.save_model_path, f'{self.model_name}.pt')

        print(model)
        lossFn = CSVLoss()
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_epochs = []
        for epoch in range(epochs):
            loss_batchs = []
            # batch = 0
            for features, labels in tqdm(dataLoader):
                features = features.to(device)
                labels[0] = labels[0].to(device)
                labels[1] = labels[1].to(device)
                y = model(features)
                loss = lossFn(y, labels)
                loss.backward()
                optim.step()
                optim.zero_grad()
                with torch.no_grad():
                    loss_batch = loss.detach().cpu().numpy()
                    loss_batchs.append(loss_batch)
                # batch += 1
                # if batch % 100 == 0:
                # tqdm.write(f'loss: epoch={epoch + 1} - batch={batch} - loss={loss_batch}')
                # if save_path:
                #     torch.save(model.state_dict(), save_path)
            loss_mean = np.mean(loss_batchs)
            print(f'{self.model_name} {epoch + 1}/{epochs} loss: {loss_mean} eta：{learning_rate}\n')
            loss_epochs.append(loss_mean)
            if save_path and epoch % 10 == 0:
                torch.save(model.state_dict(), save_path)
        return loss_epochs

    def eval(self, data_csv_path, load_model_path):
        dataset = CSVDataset(csv_path=data_csv_path, index_column=self.index_column,
                             feature_columns=self.feature_columns, label_regression=self.label_regression,
                             label_classify=self.label_classify, training=False)
        dataLoader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1)

        model = CSVModule(in_features=len(self.feature_columns), classify_sum=2)
        model = model.cpu()
        model.eval()

        if load_model_path and os.path.exists(load_model_path):
            model.load_state_dict(torch.load(load_model_path))

        eval_data = pd.DataFrame(columns=[self.label_regression, *self.label_classify])
        for features in tqdm(dataLoader):
            y_regression, y_classify, = model(features)
            y_regression = y_regression.detach().numpy()
            y_classify = torch.argmax(y_classify, dim=1).detach().numpy()

            y_regression = pd.DataFrame(y_regression, columns=[self.label_regression])
            y_classify = pd.DataFrame(y_classify, columns=[*self.label_classify], dtype=int)

            batch_data = pd.concat([y_regression, y_classify], axis=1)
            eval_data = pd.concat([eval_data, batch_data])

        eval_data = eval_data.set_index(dataset.data[self.index_column])

        return eval_data
