import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F

batch_size=64

train_data=datasets.MNIST('./mnist/',download=True, transform=transforms.ToTensor(), train=True)
test_data=datasets.MNIST('./mnist/', download=True, transform=transforms.ToTensor(), train=False)


train_loader=DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size)
test_loader=DataLoader(dataset=test_data, shuffle=True, batch_size=batch_size)


class Model(torch.nn.Module):
	
	def __init__(self):
		super(Model,self).__init__()
		self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
		self.conv2=torch.nn.Conv2d(10,20, kernel_size=5)
		self.mp=torch.nn.MaxPool2d(kernel_size=2)
		self.fc=torch.nn.Linear(320,10)

	def forward(self,x):
		in_size=x.size(0)
		x=F.relu(self.mp(self.conv1(x)))
		x=F.relu(self.mp(self.conv2(x)))
		x=x.view(in_size,-1)
		x=self.fc(x)

		return F.log_softmax(x)


model=Model()

#loss_crit=torch.nn.CrossEntropyLoss(size_average=True)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.5)


def train(epoch):
	model.train()
	for batch_idx,(data,target) in enumerate(train_loader):
		data,target=Variable(data),Variable(target)
		optimizer.zero_grad()
		output=model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
	if batch_idx % 10 == 0:
		print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
test()