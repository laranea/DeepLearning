import torch

class MLP(torch.nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = torch.nn.Linear(784, 520)
        self.l2 = torch.nn.Linear(520, 320)
        self.l3 = torch.nn.Linear(320, 240)
        self.l4 = torch.nn.Linear(240, 120)
        self.l5 = torch.nn.Linear(120, 10)

        self.ht1 = torch.nn.Hardtanh()
        self.ht2 = torch.nn.Hardtanh()
        self.ht3 = torch.nn.Hardtanh()
        self.ht4 = torch.nn.Hardtanh()

        self.bn1 = torch.nn.BatchNorm1d(520)
        self.bn2 = torch.nn.BatchNorm1d(320)
        self.bn3 = torch.nn.BatchNorm1d(240)
        self.bn4 = torch.nn.BatchNorm1d(120)


        self.logsoftmax=torch.nn.LogSoftmax()


    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = self.bn1(self.ht1(self.l1(x)))
        x = self.bn2(self.ht2(self.l2(x)))
        x = self.bn3(self.ht3(self.l3(x)))
        x = self.bn4(self.ht4(self.l4(x)))
        return self.logsoftmax(self.l5(x))
