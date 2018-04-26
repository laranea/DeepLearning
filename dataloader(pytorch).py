import torch
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader


class Diabetes(Dataset):
	
	def __init__(self):
		xy=np.loadtxt('./diabetes.csv',delimiter=',',dtype=np.float32)

		self.len=xy.shape[0]
		self.x_data=Variable(torch.from_numpy(xy[:,0:-1]))
		self.y_data=Variable(torch.from_numpy(xy[:,[-1]]))

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.len


dataset=Diabetes()

train_loader=DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # Run your training process
print(epoch, i, "inputs", inputs.data, "labels", labels.data)