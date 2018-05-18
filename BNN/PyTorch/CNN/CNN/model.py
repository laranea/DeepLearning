import torch
import torch.nn.functional as F


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(10,20, kernel_size=5)
#        self.convdrop=torch.nn.Dropout2d()
        self.ht1=torch.nn.Hardtanh()
        self.ht2=torch.nn.Hardtanh()
        self.bn1=torch.nn.BatchNorm2d(10)
        self.bn2=torch.nn.BatchNorm2d(20)
        self.fc1=torch.nn.Linear(320,50)
        self.fc2=torch.nn.Linear(50,10)
        self.ht_fc1=torch.nn.Hardtanh()
        self.bn_fc1=torch.nn.BatchNorm1d(50)
        self.mp1=torch.nn.MaxPool2d(kernel_size=2)
        self.mp2=torch.nn.MaxPool2d(kernel_size=2)



    def forward(self,x):
        in_size=x.size(0)
        x=self.conv1(x)
        x=self.mp1(x)
        x=self.bn1(x)
        x=self.ht1(x)
        x=self.conv2(x)
        x=self.mp2(x)
        x=self.bn2(x)
        x=self.ht2(x)

        x=x.view(in_size,-1)
        x=self.fc1(x)
        x=self.bn_fc1(x)
        x=self.ht_fc1(x)
        x=self.fc2(x)

        return F.log_softmax(x)
