import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


batch_size=64

train_dataset=datasets.MNIST(root='./mnist/',download=True, train=True, transform=transforms.ToTensor())
test_dataset=datasets.MNIST(root='mnist', download=True, train=False, transform=transforms.ToTensor())


train_loader=DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

class Model(torch.nn.Module):

	def __init__(self):
		super(Model,self).__init__()

		self.l1=torch.nn.Linear(784,520)
		self.l2=torch.nn.Linear(520,320)
		self.l3=torch.nn.Linear(320,640)
		self.l4=torch.nn.Linear(640,240)
		self.l5=torch.nn.Linear(240,120)
		self.l6=torch.nn.Linear(120,10)


	def forward(self,x):
		x=x.view(-1,784)
		x=F.relu(self.l1(x))
		x=F.relu(self.l2(x))
		x=F.relu(self.l3(x))
		x=F.relu(self.l4(x))
		x=F.relu(self.l5(x))
		return self.l6(x)


model=Model()

loss_crit=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)


def train(epoch):
	model.train()
	for batch_idx, (data,target) in enumerate(train_loader):
		data,target=Variable(data),Variable(target)
		optimizer.zero_grad()
		output=model(data)
		loss=loss_crit(output,target)
		loss.backward()
		optimizer.step()

		if batch_idx % 10==0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
100. * batch_idx / len(train_loader), loss.data[0]))



def test():
	model.eval()
	test_loss=0
	correct=0
	for data,target in test_loader:
		data,target=Variable(data,volatile=True),Variable(target)
		output=model(target)
		test_loss+=loss_crit(output,target).data[0]
		pred = output.data.max(1, keepdim=True)[1]
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()
	test_loss /= len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
100. * correct / len(test_loader.dataset)))


for epoch in range(1,10):
	train(epoch)
	test()
