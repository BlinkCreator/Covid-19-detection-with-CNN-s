import torch.nn as nn
import torch.nn.functional as F
import torch


# The network should inherit from the nn.Module
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 1: input channls 32: output channels, 3: kernel size, 1: stride
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=10, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)

        # Dropout layers go here
        # elf.dropout2 = nn.Dropout()
        # elf.dropout1 = nn.Dropout()

        self.fc1 = nn.Linear(192 * 1 * 1 , 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.out = nn.Linear(in_features=500, out_features=2)

    def forward(self, t):
        # Hidden conv layer1o
        t = self.conv1(t)
        t = F.relu(t)
       # print(t.shape)
        t = F.avg_pool2d(t, kernel_size=2, stride=2)  # stride size may be 1

        # Hidden conv layer2
        t = self.conv2(t)
        t = F.relu(t)
       # print(t.shape)
        t = F.avg_pool2d(t, kernel_size=2, stride=2)

        # Hidden linear layer2
       #o print(t.shape)
        #
        t = t.reshape(-1, 192 * 1 * 1) #need
        t = self.fc1(t)
        t = F.relu(t)
        #print(t.shape)

        # Hidden linear layer2
        t = self.fc2(t)
        t = F.relu(t)

        # Output layer

        t = self.out(t)
        output = F.log_softmax(t, dim = 1)
        output = output.cuda()
        return output