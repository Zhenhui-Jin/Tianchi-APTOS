import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

import pandas as pd
from tqdm import tqdm


class APTOSDataset(Dataset):
    def __init__(self, csv_path: str, img_column: str, label_names: list):
        self.data = pd.read_csv(csv_path)
        self.img_column = img_column
        self.label_names = label_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index]
        image = read_image(data[self.img_column])
        print(index, image.shape)
        label = data[self.label_names].array
        label = label.astype('float32')
        label = torch.Tensor(label)
        return image, label


class APTOSModel:
    def __init__(self, csv_path: str, img_column: str, label_names: list):
        self.dataset = APTOSDataset(csv_path, img_column, label_names)

    def fit(self, epochs=5, batch_size=16):
        dataLoader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=3)
        for epoch in range(epochs):
            print(f'{epoch + 1}/{epochs}')
            for images, labels in tqdm(dataLoader):
                pass


if __name__ == '__main__':
    # train_dataset = APTOSDataset('data/train_data_after.csv', 'ImgPath',
    #                              ['preCST', 'preIRF', 'preSRF', 'prePED', 'preHRF'])
    model = APTOSModel('data/train_data_after.csv', 'ImgPath', ['preCST', 'preIRF', 'preSRF', 'prePED', 'preHRF'])
    model.fit(batch_size=64)
