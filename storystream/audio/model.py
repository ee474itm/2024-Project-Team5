import torch.nn as nn


class M2EClassifier(nn.Module):
    def __init__(self):
        super(M2EClassifier, self).__init__()
        self.dense1 = nn.Linear(161, 75)
        self.dropout = nn.Dropout(p=0.5)
        self.dense2 = nn.Linear(75, 50)
        self.dense3 = nn.Linear(50, 50)
        self.dense4 = nn.Linear(50, 50)
        self.dense5 = nn.Linear(50, 6)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.l2_reg = 0.01

    def forward(self, x):
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))
        x = self.relu(self.dense4(x))
        x = self.softmax(self.dense5(x))
        return x
