import numpy as np
import tensorflow as tf


def sigmoid(x):
    return 1/(1+np.exp(-x))

n_input=3
n_hidden=2
n_output=1

x=tf.placeholder("float", [None,n_input])
y=tf.placeholder("float", [None,n_output])


w = {"h1": tf.Variable(tf.ones([n_input, n_hidden])),
        "out": tf.Variable(tf.ones([n_hidden,n_output]))}

b = {"b1": tf.Variable(tf.zeros([n_hidden])),
    "out": tf.Variable(tf.zeros([n_output]))}

l1= tf.add(tf.matmul(x,w["h1"]),b["b1"])
l1_activation = tf.sigmoid(l1)

out = tf.add(tf.matmul(l1_activation,w["out"]),b["out"])
out_activation= tf.sigmoid(out)

cost= tf.reduce_mean(tf.abs(tf.subtract(out_activation,y)))
optimizer=tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(cost)

x_raw=([1,2,3],[5,3,4])
y_raw=([3],[5])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pred=out_activation.eval({x:x_raw})
    print(pred)

    for epoch in range(1,10):
        sess.run(optimizer, feed_dict={x_raw,y_raw})
        #total_loss+=l
        #print("Epoch {0}: {1}".format(epoch,total_loss/n_samples))
        
