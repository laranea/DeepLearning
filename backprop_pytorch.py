import torch
import numpy as np
from torch.autograd import Variable


w=Variable(torch.Tensor([1]),requires_grad=True)
#x_data=Variable(torch.Tensor([[1,2,3]]))
#y_data=Variable(torch.Tensor([[3,4,5]]))
x_data=[1,2,3]
y_data=[2,3,4]



def loss(x,y):
	y_pred=forward(x)
	return (y_pred-y)**2

def forward(x):
	return x*w


for epoch in range(1,100):
	for x_val,y_val in zip(x_data,y_data):
		l=loss(x_val,y_val)
		l.backward()
		w.data=w.data-0.01*w.grad.data
	
#		w.grad.data=torch.Tensor([[0]])

	print("Loss:",l.data[0])