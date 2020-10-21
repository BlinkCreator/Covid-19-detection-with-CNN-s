import torch
import torch.nn as nn
import torch.nn.functional as F

# Inherits from the nn.Module
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # 1: input channels 32: output channels, 3: kernel size, 1: stride, 2, padding, 2
        # Vgg16 uses 13 convolutional layers.
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv7 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv8 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv9 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv10 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv11 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv13 = nn.Conv2d(64, 64, 3, 1, 1)

        # Utilizes 3 dropout layers to avoid overfitting
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.dropout3 = nn.Dropout2d(0.5)

        # Utilizes 3 fully connected layers
        self.fc1 = nn.Linear(64 * 2 * 2, 4096)
        self.fc2 = nn.Linear(4096, 1000)
        self.fc3 = nn.Linear(1000, 2)

    def forward(self, t):
        # 1st convolutional layer
        t = self.conv1(t)
        # ReLu function used to help with the diminishing gradient problem here and throughout
        t = F.relu(t)

        # 2nd convolutional layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, 2)
        # max_Pool Reduces the size of the matrix by taking the highest
        # value of the filter. Used here and throughout

        # 3rd convolutional layer
        t = self.conv3(t)
        t = F.relu(t)

        # 4th convolutional layer
        t = self.conv4(t)
        t = F.relu(t)

        # 5th convolutional layer
        t = F.max_pool2d(t, 2)
        t = self.conv5(t)
        t = F.relu(t)

        # 6th convolutional layer
        t = self.conv6(t)
        t = F.relu(t)

        # 7th convolutional layer
        t = self.conv7(t)
        t = F.relu(t)
        t = F.max_pool2d(t, 2)

        # 8th convolutional layer
        t = self.conv8(t)
        t = F.relu(t)

        # 9th convolutional layer
        t = self.conv9(t)
        t = F.relu(t)

        # 10th convolutional layer
        t = self.conv10(t)
        t = F.relu(t)
        t = F.max_pool2d(t, 2)

        # 11th convolutional layer
        t = self.conv11(t)
        t = F.relu(t)

        # 12th convolutional layer
        t = self.conv12(t)
        t = F.relu(t)

        # 13th convolutional layer
        t = self.conv13(t)
        t = F.relu(t)
        t = F.max_pool2d(t, 2)

        # First fully connected layer
        # print(t.shape) use this to find the shape of the matrix
        # after running in order to correctly shape the input tensor for the first linear layer
        t = self.dropout1(t)
        t = t.view(-1, 64 * 2 * 2)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.dropout2(t)

        # Second fully connected layer
        t = self.fc2(t)
        t = F.relu(t)
        t = self.dropout3(t)

        # Output layer
        t = self.fc3(t)
        output = F.log_softmax(t, dim=1)  # Normalizes CNN's output between 0 and 1

        return output
