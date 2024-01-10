import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Define the layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,padding= 2 , kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, padding= 2,kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,padding= 2, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Add three linear layers
        self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc1 = nn.Linear(in_features=256 * 8 * 8, out_features=128)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(64 , 6)

    def forward(self, x):
        # Define the forward pass
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = torch.flatten(x, 1)
        # Pass the flattened output through the three linear layers
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu5(x)
        x = self.fc3(x)

        return x

    def write_weights(self, fname):
        """ Store learned weights of CNN """
        torch.save(self.state_dict(), fname)

    def load_weights(self, fname):
        """
        Load weights from file in fname.
        The evaluation server will look for a file called checkpoint.pt
        """
        ckpt = torch.load(fname)
        self.load_state_dict(ckpt)


def get_loss_function():
    """Return the loss function to use during training. We use
       the Cross-Entropy loss for now.
    
    See https://pytorch.org/docs/stable/nn.html#loss-functions.
    """
    return nn.CrossEntropyLoss()


def get_optimizer(network, lr=0.1,  momentum=0.9 ):
    """Return the optimizer to use during training.
    network specifies the PyTorch model.

    See https://pytorch.org/docs/stable/optim.html#how-to-use-an-optimizer.
    """

    # betas=(0.9, 0.999)
    # The fist parameter here specifies the list of parameters that are
    # learnt. In our case, we want to learn all the parameters in the network
    # return optim.Adam(network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-09) 
    return optim.SGD(network.parameters(), lr=lr, momentum=momentum)

