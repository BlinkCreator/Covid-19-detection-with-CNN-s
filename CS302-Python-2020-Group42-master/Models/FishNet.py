import torch.nn as nn
import torch.nn.functional as F
import torch


# The network should inherit from the nn.Module
class FishNet(nn.Module):
    def __init__(self):
        super(FishNet, self).__init__()
        # This model is based off AlexNet. With drop out layers added and some convolution layer removed.
        # This custom model is designed to combat the drop in efficiency when teaching alexNet.

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)

        # This is where to two extra dropout layers were added.
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(384 * 3 * 3 , 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.out = nn.Linear(in_features=500, out_features=2)

    def forward(self, t):
        # Hidden conv layer1
        t = self.conv1(t)
        t = F.relu(t)  # Used to help wit the diminishing gradient problem.
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # reduces size of image
        # and uses the highest value within the filter.

        # Hidden conv layer2
        t = self.conv2(t)
        t = F.relu(t)

        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # Hidden conv layer3
        t = self.conv3(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # Hidden linear layer2
        t = self.dropout1(t)
        # print(t.shape) used this to find the shape of the matrices after running in order to correctly shape inputs.
        t = t.reshape(-1, 384 * 3 * 3)
        t = self.fc1(t)
        t = F.relu(t)

        # Hidden linear layer2
        t = self.fc2(t)
        t = self.dropout2(t)
        t = F.relu(t)

        # Output layer
        t = self.out(t)
        output = F.log_softmax(t, dim=1)# used to noralize the output values between 0 an 1.
        output = output.cuda()
        return output
