import torch.nn as nn


class APTOSLoss(nn.Module):
    def __init__(self):
        super(APTOSLoss, self).__init__()
        reduction = 'sum'
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.bce_log_loss = nn.BCEWithLogitsLoss(reduction=reduction)  # 将多标签的二分类loss相加

    def forward(self, inputs: tuple, targets: tuple):
        y_regression, y_classify = inputs
        regression_labels, classify_labels = targets
        mse_loss = self.mse_loss(y_regression, regression_labels)
        bce_log_loss = self.bce_log_loss(y_classify, classify_labels)
        # print(mse_loss, bce_log_loss, mse_loss + bce_log_loss)
        return mse_loss + bce_log_loss
