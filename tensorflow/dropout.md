#tensorflow:dropout
**我们都知道dropout对于防止过拟合效果不错**
**drop一般用在全连接的部分，卷积部分不会用到dropout,输出曾也不会使用dropout，适用范围[输入，输出)**
1.tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
2.tf.nn.rnn_cell.DropoutWrapper(rnn_cell, input_keep_prob=1.0, output_keep_prob=1.0)
##普通dropout
```python
def dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
#x: 输入
#keep_prob: 名字代表的意思
#return：包装了dropout的x。训练的时候用，test的时候就不需要dropout了
#例：
w = tf.get_variable("w1",shape=[size, out_size])
x = tf.placeholder(tf.float32, shape=[batch_size, size])
x = tf.nn.dropout(x, keep_prob=0.5)
y = tf.matmul(x,w)
```
##rnn中的dropout
```python
def rnn_cell.DropoutWrapper(rnn_cell, input_keep_prob=1.0, output_keep_prob=1.0):
#例
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
#经过dropout包装的lstm_cell就出来了
```
