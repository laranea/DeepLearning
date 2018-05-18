import torch

class MLP(torch.nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = torch.nn.Linear(784, 2048)
        self.l2 = torch.nn.Linear(2048, 2048)
        self.l3 = torch.nn.Linear(2048, 2048)
        self.l4 = torch.nn.Linear(2048, 10)


        self.ht1 = torch.nn.Hardtanh()
        self.ht2 = torch.nn.Hardtanh()
        self.ht3 = torch.nn.Hardtanh()


        self.bn1 = torch.nn.BatchNorm1d(2048)
        self.bn2 = torch.nn.BatchNorm1d(2048)
        self.bn3 = torch.nn.BatchNorm1d(2048)



        self.logsoftmax=torch.nn.LogSoftmax()


    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = self.bn1(self.ht1(self.l1(x)))
        x = self.bn2(self.ht2(self.l2(x)))
        x = self.bn3(self.ht3(self.l3(x)))
        return self.logsoftmax(self.l4(x))
