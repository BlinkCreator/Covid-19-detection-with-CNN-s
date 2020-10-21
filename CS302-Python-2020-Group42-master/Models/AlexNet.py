import torch.nn as nn
import torch.nn.functional as F


# Inherits from the nn.Module
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # 1: input channels 32: output channels, 3: kernel size, 1: stride, 2, padding, 2
        # Alex net utilizes 5 convolutional layers.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Dropout layers go here but because AlexNet does not utilize dropout layers there are none.

        self.fc1 = nn.Linear(256 * 1 * 1, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.out = nn.Linear(in_features=500, out_features=2)

    def forward(self, t):
        # Hidden conv layer1
        t = self.conv1(t)
        # Relu function used to help with the diminishing gradient problem here and throughout
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # max_Pool Reduces the size of the matrix by taking the highest
        # value of the filter. Used here and throughout

        # Hidden conv layer2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # Hidden conv layer3
        t = self.conv3(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # Hidden conv layer4
        t = self.conv4(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=1)

        # Hidden conv layer5
        t = self.conv5(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=1)

        # Hidden linear layer1
        # print(t.shape) use this to find the shape of the matrix
        # after running in order to correctly shape the input tensor for the first linear layer
        t = t.reshape(-1, 256 * 1 * 1)  # Reshapes the tensor for the fully connected layer.
        t = self.fc1(t)
        t = F.relu(t)
        # print(t.shape)

        # Hidden linear layer2
        t = self.fc2(t)
        t = F.relu(t)

        # Output layer
        t = self.out(t)
        output = F.log_softmax(t, dim=1)  # Normalizes CNN's output between 0 and 1
        return output
