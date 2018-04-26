import numpy as np

x_data=[1,2,3]
y_data=[2,3,4]
w=1

def forward(x):
	return x*w

def loss(x,y):
	y_pred=forward(x)
	return (y_pred-y)**2

#for w in np.arange(0,4,1):
#	print("w",w)
#	l_sum=0
#	for x_val,y_val in zip(x_data,y_data):
#		y_pred_val=forward(x_val)
#		l=loss(y_pred_val,y_val)
#		l_sum+=l
#		print(x_val,y_val,y_pred_val,l)

#	print("MSE:",l_sum/3)



def gradient_descent(x,y):
	return 2*x*(x*w-y)


for epoch in range(1,100):
	for x_val,y_val in zip(x_data,y_data):
		grad=gradient_descent(x_val,y_val)
		w=w-0.01*grad
		l=loss(x_val,y_val)
	print("Epoch:",epoch,"\t","w",w,"\t","loss:",l)