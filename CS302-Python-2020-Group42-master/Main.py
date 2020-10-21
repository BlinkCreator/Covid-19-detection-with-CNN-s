# functions to show an image
import torch
import torchvision
import torch.nn.functional as F
from Models.FishNet import FishNet
from Models.AlexNet import AlexNet
from Models.VGG16 import VGG16
from Models.LeNet5 import LeNet5
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn import metrics
import torch.optim as optim
import numpy as np

training_loss = []
testing_accuracy = []
testing_loss = []

# Function to evaluate and display Test accuracy and loss as well as training loss
def Evaluation(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data = torch.squeeze(data)
            data, target = data.to(device), target.to(device)
            y_true.append(target.tolist())
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            prediction = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            temp_prediction = prediction
            temp_prediction = temp_prediction.view(25)
            y_pred.append(temp_prediction.tolist())
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    y_true_flattened = [y for x in y_true for y in x]
    y_pred_flattened = [y for x in y_pred for y in x]

    # Prints the confusion matrix and classification report
    print('------------Confusion Matrix------------')
    print(metrics.confusion_matrix(y_true_flattened, y_pred_flattened, labels=[0, 1]))
    print('------------Classification Report------------')
    print(metrics.classification_report(y_true_flattened, y_pred_flattened, labels=[0, 1]))
    plt.plot(training_loss)
    plt.ylabel('Loss(training)')
    plt.show()
    plt.plot(testing_accuracy, label='Accuracy(testing)')
    plt.plot(testing_loss, label='Loss(testing)')
    plt.legend()
    plt.show()


# Function for training the neural network
def train_cnn(log_interval, model, device, train_loader, optimizer, epoch):
    #
    iterations = 0
    epoch_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output = model(data)
        loss = F.nll_loss(output, target)
        epoch_loss += loss
        iterations += 1
        loss.backward(); optimizer.step()
        # prints the loss and percentage done of each epoch
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    training_loss.append(epoch_loss/iterations)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    results = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = torch.squeeze(data)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            prediction = output.argmax(dim=1, keepdim=True)  # get the index of the2 max log-probability
            results += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    testing_accuracy.append(results/len(test_loader.dataset))
    testing_loss.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, results, len(test_loader.dataset),
        100. * results / len(test_loader.dataset)))


def main():
    epoches = 10
    gamma = 0.7
    log_interval = 10
    torch.manual_seed(1)
    save_model = True

    # Check whether you can use Cuda
    use_cuda = torch.cuda.is_available()
    # Uses Cuda if you can
    device = torch.device("cuda" if use_cuda else "cpu")

    # Torchvision
    # Use data predefined loader
    # Pre-processing by using the transform.Compose
    # divide into batches
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # loads dataset

    # Sets up a function that Transforms and normalizes our data
    transform = transforms.Compose(
        [transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Loads the training set of data with a batch size 32
    trainset = torchvision.datasets.ImageFolder(root='./Dataset/Train', transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=0)

    # Loads the training set of data with a batch size 32
    testset = torchvision.datasets.ImageFolder(root='./Dataset/test', transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=25,
                                             shuffle=False, num_workers=0)
    # classes = ('Normal', 'COVID-19')

    # Randomizes training images
    #dataiter = iter(train_loader)
    #images, labels = dataiter.next()
    #img = torchvision.utils.make_grid(images)
    #imsave(img)

    # Build your network and run

    # Custom Interface to select model
    mdlNum = input(
        "\n\t\033[95m[CHOSE MODEL TO RUN]\033[0m\033[94m\n\t0 => FishNet\n\t1 => AlexNet\n\t2 => VGG16 (Default)\n\t3 => LeNet5\n\033[0m\033[92m >> ")

    if mdlNum == "0":
        model = FishNet().to(device)
        print("\n\t\033[95mStarting FishNet\033[93m")
    elif mdlNum == "1":
        model = AlexNet().to(device)
        print("\n\t\033[95mStarting AlexNet\033[93m")
    elif mdlNum == "2":
        model = VGG16().to(device)
        print("\n\t\033[95mStarting VGG16\033[93m")
    elif mdlNum == "3":
        model = LeNet5().to(device)
        print("\n\t\033[95mStarting LeNet5\033[93m")
    else:
        print("\n\t\033[95mStarting VGG16 (Default)\033[93m")
        model = VGG16().to(device)

    # Sets the learning model and learning rate of 0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(1, epoches + 1):

        train_cnn(log_interval, model, device, train_loader, optimizer, epoch)

        test(model, device, test_loader)
        scheduler.step()

    Evaluation(model, device, test_loader)

    if save_model:
        torch.save(model.state_dict(), "./results/mnist_cnn.pt")


if __name__ == '__main__':
    main()

