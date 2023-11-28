import torch.nn.functional as F
import torch.nn as nn

#Define the models
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in,128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, dim_out)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        #x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        x = self.activation(x)
        return x
    
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_drop = nn.Dropout2d(p=0.1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, num_classes)
        self.activation = nn.LogSoftmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.output(x)
        x = self.activation(x)
        return x
    
class CNN2(nn.Module):
    def __init__(self, num_classes):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3) #in-ch, out-ch, kernel_size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.output = nn.Linear(64, num_classes)
        self.activation = nn.LogSoftmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        x = self.activation(x)
        return x
    
class CNNFashion(nn.Module):
    def __init__(self, num_classes):
        super(CNNFashion, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3) #in-ch, out-ch, kernel_size, 28x28
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3) 
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.output = nn.Linear(64, num_classes)
        self.activation = nn.LogSoftmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #28x28 -> 26x26 -> 13x13
        x = self.pool(F.relu(self.conv2(x))) #13x13 -> 11x11 -> 5x5
        #x = self.pool(F.relu(self.conv3(x))) #5x5 -> 3x3 -> 1x1
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        x = self.activation(x)
        return x