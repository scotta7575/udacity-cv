## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # From the paper it WOULD BE helpful to have 3 levels of convolution, each halving the number of outputs
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1_drop = nn.Dropout(p=0.5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # second conv layer: 32 inputs, ??? outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (13-3)/1 +1 = 11
        # the output tensor will have dimensions: (20, 11, 11)
        # after another pool layer this becomes (20, 5, 5); 5.5 is rounded down
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # 20 outputs * the 5*5 filtered/pooled map size
        linear_input = 80000
#         next_level = int(linear_input/2)
        next_level = 1000
        self.fc1 = nn.Linear(linear_input, next_level)
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        
        # finally, create 68 output channels (for the 68 keypoints)
        self.fc2 = nn.Linear(next_level, 68*2)
        

        
    def forward(self, x):
        print_network = False
        if print_network:
            print('start forward', x.shape)
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1_drop(x)
        if print_network:
            print('self.pool(F.relu(self.conv1(x)))', x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        if print_network:
            print('self.pool(F.relu(self.conv2(x)))', x.shape)
        x = self.pool(F.relu(self.conv3(x)))
        if print_network:
            print('self.pool(F.relu(self.conv3(x)))', x.shape)
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        if print_network:
            print('x.view(x.size(0), -1)', x.shape)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        if print_network:
            print('F.relu(self.fc1(x))', x.shape)
        x = self.fc1_drop(x)
        x = self.fc2(x)        
        if print_network:
            print('self.fc2(x)', x.shape)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
