import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3, stride=4, padding=0, bias=True)
        self.max_pool_1 = torch.nn.MaxPool1d(kernel_size=16)
        self.conv2 = torch.nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, stride=2, padding=0, bias=True)
        self.max_pool_2 = torch.nn.MaxPool1d(kernel_size=8)
        self.conv3 = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.max_pool_3 = torch.nn.MaxPool1d(kernel_size=2)
        self.fc1 = torch.nn.Linear(480, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = F.relu(self.conv3(x))
        x = self.max_pool_3(x)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class EpicNetwork(nn.Module):
    def __init__(self):
        super(EpicNetwork, self).__init__()
        self.conv = nn.Conv1d(      # input: (64000, 1, 1)
                in_channels=1,
                out_channels=128,
                kernel_size=3,
                stride=3,
                padding=4,         # output: (21,336, 128, 1)
        )

        self.conv1 = nn.Sequential(         # input shape (21,336, 128, 1)
            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=0,
            ),                              # output shape (21,336, 128, 1)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),  # output shape (7,112, 128, 1)
        )
        self.conv2 = nn.Sequential(         # input shape (7,112, 128, 1)
            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=0,
            ),                              # output shape (7,112, 128, 1)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),  # output shape (2,371, 128, 1)
        )
        self.conv3 = nn.Sequential(         # input shape (2,731, 128, 1)
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0,
            ),                              # output shape (2,731, 256, 1)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),  # output shape (791, 256, 1)
        )
        self.conv4 = nn.Sequential(         # input shape (791, 256, 1)
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0,
            ),                              # output shape (791, 256, 1)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),  # output shape (264, 256, 1)
        )
        self.conv5 = nn.Sequential(         # input shape (264, 256, 1)
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0,
            ),                              # output shape (264, 256, 1)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),  # output shape (88, 256, 1)
        )
        self.conv6 = nn.Sequential(         # input shape (88, 256, 1)
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0,
            ),                              # output shape (88, 256, 1)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),  # output shape (30, 256, 1)
        )
        self.conv7 = nn.Sequential(         # input shape (30, 128, 1)
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0,
            ),                              # output shape (30, 256, 1)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),  # output shape (10, 256, 1)
        )
        self.conv8 = nn.Sequential(         # input shape (10, 256, 1)
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0,
            ),                              # output shape (10, 256, 1)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),  # output shape (4, 256, 1)
        )
        self.conv9 = nn.Sequential(         # input shape (4, 256, 1)
            nn.Conv1d(
                in_channels=256,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
            ),                              # output shape (4, 512, 1)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),  # output shape (2, 512, 1)
        )
        self.conv10 = nn.Sequential(         # input shape (2, 512, 1)
            nn.Conv1d(
                in_channels=512,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
            ),                              # output shape (2, 512, 1)
            nn.ReLU(),              # output shape (2, 512, 1)
        )
        self.drop = F.dropout
        self.fc0 = nn.Linear(512, 128)   # fully connected layer, output 10 classes
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        x = self.fc1(x)
        return x
