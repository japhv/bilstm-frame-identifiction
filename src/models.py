import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.conv0 = nn.Conv1d(1, 3, 1)
        self.conv1 = nn.Conv1d(3, 6, 5)
        self.conv2 = nn.Conv1d(6, 16, 5)
        self.conv3 = nn.Conv1d(16, 32, 5)
        self.fc0 = nn.Linear(17024, 1024)
        self.fc1 = nn.Linear(1024, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv0(x))
        out = F.max_pool1d(out, 2)
        out = F.relu(self.conv1(out))
        out = F.max_pool1d(out, 3)
        out = F.relu(self.conv2(out))
        out = F.max_pool1d(out, 4)
        out = F.relu(self.conv3(out))
        out = F.max_pool1d(out, 5)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc0(out))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return F.log_softmax(out, dim=1)


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
                in_channels=128,
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
                kernel_size=3,
                stride=1,
                padding=0,
            ),                              # output shape (4, 512, 1)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),  # output shape (2, 512, 1)
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
        self.out = nn.Linear(2 * 512 * 1, 10)   # fully connected layer, output 10 classes

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
        output = self.out(x)
        return output, x
