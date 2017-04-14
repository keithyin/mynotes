# conv_transpose

`deconv`解卷积，实际是叫做`conv_transpose`, `conv_transpose`实际是卷积的一个逆向过程，`tf` 中， 编写`conv_transpose`代码的时候，心中想着一个正向的卷积过程会很有帮助。

想象一下我们有一个正向卷积:
input_shape = [1,5,5,3]
kernel_shape=[2,2,3,1]
strides=[1,2,2,1]
padding = "SAME"
那么，卷积激活后，我们会得到 x(就是上面代码的x)。那么，我们已知x，要想得到input_shape 形状的 tensor，我们应该如何使用`conv2d_transpose`函数呢？
就用下面的代码
```python
import tensorflow as tf
tf.set_random_seed(1)
x = tf.random_normal(shape=[1,3,3,1])
#正向卷积的kernel的模样
kernel = tf.random_normal(shape=[2,2,3,1])

# strides 和padding也是假想中 正向卷积的模样。当然，x是正向卷积后的模样
y = tf.nn.conv2d_transpose(x,kernel,output_shape=[1,5,5,3],
    strides=[1,2,2,1],padding="SAME")
# 在这里，output_shape=[1,6,6,3]也可以，考虑正向过程，[1,6,6,3]
# 通过kernel_shape:[2,2,3,1],strides:[1,2,2,1]也可以
# 获得x_shape:[1,3,3,1]
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

print(y.eval(session=sess))
```
