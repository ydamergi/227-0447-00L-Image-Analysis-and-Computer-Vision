import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BasicBlock(nn.Module):
    """
    Standard ResNet block with two 3x3 convolutions and a shortcut connection.
    """
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """
    ResNet model by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun
    https://arxiv.org/pdf/1512.03385.pdf
    """
    def __init__(self, block, num_blocks, num_classes=5):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Freeze layers before the last one
        for param in self.parameters():
            param.requires_grad = False
        for param in self.layer4.parameters():
            param.requires_grad = True
        
        # Replace the last linear layer
        self.linear = nn.Sequential(
            nn.Linear(512*block.expansion, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def generate_resnet(num_classes=10, num_blocks=2):
    """
    Generate a ResNet model with 4 layers.
    """
    # if num_blocks is a single number, use the same number of blocks for all layers
    if isinstance(num_blocks, int):
        num_blocks = [num_blocks] * 4
    
    assert len(num_blocks) == 4, "num_blocks should have 4 elements."
    return ResNet(BasicBlock, num_blocks, num_classes=num_classes)

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


def get_optimizer(network, lr=0.1 ):
    """Return the optimizer to use during training.
    network specifies the PyTorch model.

    See https://pytorch.org/docs/stable/optim.html#how-to-use-an-optimizer.
    """

    # betas=(0.9, 0.999)
    # The fist parameter here specifies the list of parameters that are
    # learnt. In our case, we want to learn all the parameters in the network
    #return optim.SGD(network.parameters(), lr=lr, momentum=momentum)
    return optim.Adam(network.parameters(), lr, betas=(0.9, 0.999), eps=1e-07) 
    

