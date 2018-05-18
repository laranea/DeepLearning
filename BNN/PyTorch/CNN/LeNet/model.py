import torch
import torch.nn.functional as F


## Model

class LeNet(torch.nn.Module):

    def __init__(self):
        super(LeNet,self).__init__()

        self.convnet=torch.nn.Sequential(
                        torch.nn.Conv2d(1,6,kernel_size=(5,5)),
                        torch.nn.ReLU(),
                        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                        torch.nn.Conv2d(6, 16, kernel_size=(5, 5)),
                        torch.nn.ReLU(),
                        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                        torch.nn.Conv2d(16, 120, kernel_size=(5, 5)),
                        torch.nn.ReLU()
                        )

        self.fc=torch.nn.Sequential(
                    torch.nn.Linear(120, 84),
                    torch.nn.ReLU(),
                    torch.nn.Linear(84, 10),
                    torch.nn.LogSoftmax()
                    )

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(-1, 120)
        output = self.fc(output)
        return output
