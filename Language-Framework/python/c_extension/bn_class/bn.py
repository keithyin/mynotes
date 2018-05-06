import tensorflow as tf
from tensorflow.contrib import slim

tf.gradients()

class BatchNormalization(object):
    pass


slim.conv2d()


def bn(inputs, istraining=True):
    mean, var = tf.nn.moments(inputs, axes=[0, 1, 2])
    ema = tf.train.ExponentialMovingAverage(decay=.9)
    with tf.control_dependencies([mean, var]):
        update_ema = ema.apply(var_list=[mean, var]) # the returned op can be used to update
