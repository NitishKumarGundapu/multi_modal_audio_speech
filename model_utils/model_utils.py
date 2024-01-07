import torch as t
import torch.nn as nn
import torch.nn.functional as F

#tried from scrtach . this is MADNESS -> NO, THIS IS SPARTAAAA
class EmotionCNN(nn.Module):
    def softmax(self,x):
        e_x = t.exp(x - t.max(x))
        return e_x / e_x.sum()
    
    def __init__(self, num_classes,in_channels):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv1d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv1d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv1d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(12 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(t.relu(self.conv1(x)))
        x = self.pool(t.relu(self.conv2(x)))
        x = self.pool(t.relu(self.conv3(x)))
        x = self.pool(t.relu(self.conv4(x)))
        x = x.view(-1, 12 * 32 * 32)
        x = t.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x)
        return x
    

