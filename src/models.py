import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3, stride=2, padding=0, bias=True)
        self.max_pool_1 = torch.nn.MaxPool1d(kernel_size=16)
        self.conv2 = torch.nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, stride=2, padding=0, bias=True)
        self.max_pool_2 = torch.nn.MaxPool1d(kernel_size=8)
        self.conv3 = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.max_pool_3 = torch.nn.MaxPool1d(kernel_size=2)
        self.fc1 = torch.nn.Linear(448, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 10)
        self.drop = F.dropout

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = F.relu(self.conv3(x))
        x = self.max_pool_3(x)
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class EpicNetwork(nn.Module):
    def __init__(self):
        super(EpicNetwork, self).__init__()
        self.conv1 = nn.Conv1d(
                in_channels=1,
                out_channels=128,
                kernel_size=3,
                stride=3,
                padding=4,
                # dilation=2
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0,
                # dilation=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0,
                # dilation=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0,
                # dilation=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )
        self.fc0 = nn.Linear(512, 128)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = F.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        x = self.fc1(x)
        return x
