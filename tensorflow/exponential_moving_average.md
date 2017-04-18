# ExponentialMovingAverage
```python
# Create variables.
var0 = tf.Variable(...)
var1 = tf.Variable(...)
# ... use the variables to build a training model...
...
# Create an op that applies the optimizer.  This is what we usually
# would use as a training op.
opt_op = opt.minimize(my_loss, [var0, var1])

# Create an ExponentialMovingAverage object
ema = tf.train.ExponentialMovingAverage(decay=0.9999)

# Create the shadow variables, and add ops to maintain moving averages
# of var0 and var1.
maintain_averages_op = ema.apply([var0, var1])

# Create an op that will update the moving averages after each training
# step.  This is what we will use in place of the usual training op.
with tf.control_dependencies([opt_op]):#  ？？？？？
    training_op = tf.group(maintain_averages_op)

...train the model by running training_op...
```


**参考资料**
[https://en.wikipedia.org/wiki/Moving_average](https://en.wikipedia.org/wiki/Moving_average)
[https://www.tensorflow.org/api_docs/python/train/moving_averages#ExponentialMovingAverage](https://www.tensorflow.org/api_docs/python/train/moving_averages#ExponentialMovingAverage)
