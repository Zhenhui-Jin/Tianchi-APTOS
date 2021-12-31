import torch.nn as nn


class BasicConv2d(nn.Module):

    def __init__(self, in_features, out_features, kernel_size: int = 3, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv2d(in_features, out_features,
                      kernel_size=(kernel_size, kernel_size),
                      stride=(stride, stride),
                      padding=padding,
                      dilation=(dilation, dilation)),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class Block(nn.Module):
    def __init__(self, nin, kernel_size: int = 3, stride: int = 1):
        super(Block, self).__init__()
        self.layers = nn.ModuleList(
            [BasicConv2d(nin, nin, kernel_size, stride, 2 ** i) for i in range(5)]
        )

    def forward(self, x):
        for layer in self.layers:
            x1 = layer(x)  # 扩张率逐步增大的卷积神经网络
            x = x1 + x  # 残差网络
        return x


class APTOSModule(nn.Module):
    def __init__(self, input_shape=(3, 500, 764), out_features=64, classify_sum: int = 4):
        super(APTOSModule, self).__init__()
        self.layers = nn.Sequential(
            BasicConv2d(input_shape[0], 16),
            BasicConv2d(16, 16, stride=2),
            BasicConv2d(16, 16, stride=2),
            BasicConv2d(16, 32, stride=2),
            BasicConv2d(32, 32, stride=2),
            BasicConv2d(32, 32, stride=2),
            BasicConv2d(32, 64, stride=2),
            BasicConv2d(64, 64, stride=2),
            BasicConv2d(64, out_features, stride=2)
        )
        conv2d_count = 8
        H = int(input_shape[1] // (2 ** conv2d_count)) + 1
        W = int(input_shape[2] // (2 ** conv2d_count)) + 1
        in_features = out_features * H * W

        self.regression = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, int(in_features / 2)),
            nn.ReLU(),
            nn.Linear(int(in_features / 2), int(in_features / 4)),
            nn.ReLU(),
            nn.Linear(int(in_features / 4), 1)
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, int(in_features / 2)),
            nn.ReLU(),
            nn.Linear(int(in_features / 2), int(in_features / 4)),
            nn.ReLU(),
            nn.Linear(int(in_features / 4), classify_sum),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        y_regression = self.regression(x)
        y_classify = self.classify(x)
        return y_regression, y_classify


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
            nn.Linear(4, classify_sum)
        )

    def forward(self, x):
        x = self.layers(x)
        y_regression = self.regression(x)
        y_classify = self.classify(x)
        return y_regression, y_classify
