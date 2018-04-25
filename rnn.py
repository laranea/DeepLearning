import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM


data=[[i for i in range(100)]]
data=np.array(data,dtype=float)
target=[[i for i in range(1,101)]]
target=np.array(target,dtype=float)

data=data.reshape((1,1,100))
target=target.reshape((1,1,100))

## Test Data

x_test=[[i for i in range(50,150)]]
x_test=np.array(x_test,dtype=float).reshape(1,1,100)
y_test=[[i for i in range(51,151)]]
y_test=np.array(y_test,dtype=float).reshape(1,1,100)


model=Sequential()
model.add(LSTM(100, input_shape=(1,100), return_sequences=True))
model.add(Dense(100))
model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['accuracy'])
model.fit(data,target,nb_epoch=1000,batch_size=1,verbose=2, validation_data=(x_test,y_test))


predict=model.predict(x_test)