import torch
from torch.autograd import Variable
import torch.nn.functional as F


x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]]))
y_data = Variable(torch.Tensor([[0.], [0.], [1.], [1.]]))



class Model(torch.nn.Module):
	
	def __init__(self):
		super(Model,self).__init__()
		self.linear=torch.nn.Linear(1,1)

	def forward(self,x):
		y_pred=F.sigmoid(self.linear(x))
		return y_pred


model=Model()

loss_crit=torch.nn.BCELoss(size_average=True)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)



for epoch in range(1000):
	y_pred=model(x_data)
	loss=loss_crit(y_pred,y_data)

	print(epoch,loss.data[0])

	loss.backward()
	optimizer.step()



#testing

test=Variable(torch.Tensor([[1.0]]))
print("predict 1 hour ", 1.0, model(test).data[0][0] > 0.5)

test1 = Variable(torch.Tensor([[7.0]]))
print("predict 7 hours", 7.0, model(test1).data[0][0] > 0.5)