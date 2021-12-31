import torch.nn as nn


class ImageLoss(nn.Module):
    def __init__(self):
        super(ImageLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_log_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs: tuple, targets: tuple):
        y_regression, y_classify = inputs
        regression_labels, classify_labels = targets
        mse_loss = self.mse_loss(y_regression, regression_labels)
        bce_log_loss = self.bce_log_loss(y_classify, classify_labels)
        return mse_loss + bce_log_loss * 2


class CSVLoss(nn.Module):
    def __init__(self):
        super(CSVLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs: tuple, targets: tuple):
        y_regression, y_classify = inputs
        regression_labels, classify_labels = targets
        classify_labels = classify_labels.squeeze()
        classify_labels = classify_labels.long()
        mse_loss = self.mse_loss(y_regression, regression_labels)
        ce_loss = self.ce_loss(y_classify, classify_labels)
        return mse_loss + ce_loss
