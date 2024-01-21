import torch
import torch.nn as nn

class cnn_chestxray28(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        network = []
        # first conv
        network.append(
            nn.Conv2d(
                in_channels=1, out_channels=14, kernel_size=(3, 3), padding=0
            )
        )
        network.append(nn.ReLU())

        # second conv
        network.append(
            nn.Conv2d(
                in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0
            )
        )
        network.append(nn.ReLU())

        # first BN
        network.append(nn.BatchNorm2d(
            num_features=14, affine=False
        ))

        # first pooling
        network.append(nn.MaxPool2d(
            kernel_size=(2, 2)
        ))

        # dropout
        network.append(
            nn.Dropout(p=0.25)
        )

        # third conv
        network.append(
            nn.Conv2d(
                in_channels=14, out_channels=28, kernel_size=(3, 3), padding=0
            )
        )
        network.append(nn.ReLU())

        # fourth conv
        network.append(
            nn.Conv2d(
                in_channels=28, out_channels=28, kernel_size=(3, 3), padding=0
            )
        )
        network.append(nn.ReLU())

        # second BN
        network.append(nn.BatchNorm2d(
            num_features=28, affine=False
        ))

        # second pooling
        network.append(nn.MaxPool2d(
            kernel_size=(2, 2)
        ))

        # dropout
        network.append(
            nn.Dropout(p=0.25)
        )

        # Flatten
        network.append(
            nn.Flatten()
        )

        # Dense
        network.append(
            nn.Linear(
                in_features=448, out_features=256
            )
        )
        network.append(nn.ReLU())

        # BN
        network.append(nn.BatchNorm1d(
            num_features=256, affine=False
        ))

        self.network = nn.ModuleList(network)

        # dropout
        network.append(
            nn.Dropout(p=0.4)
        )

        # Dense
        network.append(
            nn.Linear(
                in_features=256, out_features=11
            )
        )
        network.append(nn.Softmax())

        self.network = nn.ModuleList(network)

    def forward(self, x):
        for (i, layer) in enumerate(self.network):
            x = layer(x)
            #print(i, layer, x.size())
        return x

class cnn_chestxray32(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        network = []
        # first conv
        network.append(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0
            )
        )
        network.append(nn.ReLU())

        # second conv
        network.append(
            nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0
            )
        )
        network.append(nn.ReLU())

        # first BN
        network.append(nn.BatchNorm2d(
            num_features=16, affine=False
        ))

        # first pooling
        network.append(nn.MaxPool2d(
            kernel_size=(2, 2)
        ))

        # dropout
        network.append(
            nn.Dropout(p=0.25)
        )

        # third conv
        network.append(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0
            )
        )
        network.append(nn.ReLU())

        # fourth conv
        network.append(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0
            )
        )
        network.append(nn.ReLU())

        # second BN
        network.append(nn.BatchNorm2d(
            num_features=32, affine=False
        ))

        # second pooling
        network.append(nn.MaxPool2d(
            kernel_size=(2, 2)
        ))

        # dropout
        network.append(
            nn.Dropout(p=0.25)
        )

        # Flatten
        network.append(
            nn.Flatten()
        )

        # Dense
        network.append(
            nn.Linear(
                in_features=800, out_features=256
            )
        )
        network.append(nn.ReLU())

        # BN
        network.append(nn.BatchNorm1d(
            num_features=256, affine=False
        ))

        self.network = nn.ModuleList(network)

        # dropout
        network.append(
            nn.Dropout(p=0.4)
        )

        # Dense
        network.append(
            nn.Linear(
                in_features=256, out_features=2
            )
        )
        network.append(nn.Softmax())

        self.network = nn.ModuleList(network)

    def forward(self, x):
        for (i, layer) in enumerate(self.network):
            x = layer(x)
            #print(i, layer, x.size())
        return x

class cnn_chestxray28_1(nn.Module):
    def __init__(self, num_features=2) -> None:
        super().__init__()
        self.num_features = num_features
        network = []
        # first conv
        network.append(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0
            )
        )
        network.append(nn.ReLU())

        # second conv
        network.append(
            nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0
            )
        )
        network.append(nn.ReLU())

        # first BN
        network.append(nn.BatchNorm2d(
            num_features=16, affine=False
        ))

        # first pooling
        network.append(nn.MaxPool2d(
            kernel_size=(2, 2)
        ))

        # dropout
        network.append(
            nn.Dropout(p=0.25)
        )

        # third conv
        network.append(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0
            )
        )
        network.append(nn.ReLU())

        # fourth conv
        network.append(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0
            )
        )
        network.append(nn.ReLU())

        # second BN
        network.append(nn.BatchNorm2d(
            num_features=32, affine=False
        ))

        # second pooling
        network.append(nn.MaxPool2d(
            kernel_size=(2, 2)
        ))

        # dropout
        network.append(
            nn.Dropout(p=0.25)
        )

        # Flatten
        network.append(
            nn.Flatten()
        )

        # Dense
        network.append(
            nn.Linear(
                in_features=512, out_features=256
            )
        )
        network.append(nn.ReLU())

        # BN
        network.append(nn.BatchNorm1d(
            num_features=256, affine=False
        ))

        self.network = nn.ModuleList(network)

        # dropout
        network.append(
            nn.Dropout(p=0.4)
        )

        # Dense
        network.append(
            nn.Linear(
                in_features=256, out_features=self.num_features
            )
        )
        network.append(nn.Softmax())

        self.network = nn.ModuleList(network)

    def forward(self, x):
        for (i, layer) in enumerate(self.network):
            x = layer(x)
            #print(i, layer, x.size())
        return x

if __name__ == "__main__":
    model = cnn_chestxray28()
    x = torch.rand(3, 1, 28, 28) # pytorch: [N, C, H, W]
    result = model(x)
    #print(len(result))
