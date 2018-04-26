import torch
from torch.autograd import Variable

x_data=Variable(torch.Tensor([[1],[2],[3]]))
y_data=Variable(torch.Tensor([[2],[3],[4]]))

class Model(torch.nn.Module):
	def __init__(self):
		super(Model,self).__init__()
		self.linear=torch.nn.Linear(1,1)  # input is one and output is one

	def forward(self, x):
		y_pred=self.linear(x)
		return y_pred


model=Model()


loss_criterion=torch.nn.MSELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)


for epoch in range(100):
	y_pred=model(x_data)
	l=loss_criterion(y_pred,y_data)
	print(epoch,l.data[0],y_pred)

#	optimizer.zero()
	l.backward()
	optimizer.step()

# Testing

test=Variable(torch.Tensor([[4]]))
print(model.forward(test).data[0][0])