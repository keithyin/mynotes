#LSTM & GRU
##基本LSTM
**tensorflow提供了LSTM实现的一个basic版本，不包含lstm的一些高级扩展，同时也提供了一个标准接口，其中包含了lstm的扩展。分别为：tf.nn.rnn_cell.BasicLSTMCell(), tf.nn.rnn_cell.LSTMCell()**
###LSTM的结构
盗用一下[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)上的图
![图一](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)
![图二](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png)
![图三](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png)
**tensorflow中的BasicLSTMCell()是完全按照这个结构进行设计的**
```python
#tf.nn.rnn_cell.BasicLSTMCell(num_units, forget_bias, input_size, state_is_tupe=Flase, activation=tanh)
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, forget_bias=1.0, input_size=None, state_is_tupe=Flase, activation=tanh)
#num_units:图一中ht的维数，如果num_units=10,那么ht就是10维行向量
#forget_bias：还不清楚这个是干嘛的
#input_size:[batch_size, max_time, size]。假设要输入一句话，这句话的长度是不固定的，max_time就代表最长的那句话是多长，size表示你打算用多长的向量代表一个word，即embedding_size（embedding_size和size的值不一定要一样）
#state_is_tuple:true的话，返回的状态是一个tuple:(c=array([[]]), h=array([[]]):其中c代表Ct的最后时间的输出，h代表Ht最后时间的输出，h是等于最后一个时间的output的
#图三向上指的ht称为output
#此函数返回一个lstm_cell，即图一中的一个A
```
**如果你想要设计一个多层的LSTM网络，你就会用到tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=False),这里多层的意思上向上堆叠，而不是按时间展开**
```python
lstm_cell = tf.nn.rnn_cell.MultiRNNCells(cells, state_is_tuple=False)
#cells:一个cell列表，将列表中的cell一个个堆叠起来，如果使用cells=[cell]*4的话，就是四曾，每层cell输入输出结构相同
#如果state_is_tuple:则返回的是 n-tuple，其中n=len(cells): tuple:(c=[batch_size, num_units], h=[batch_size,num_units])
```
**这是，网络已经搭好了，tensorflow提供了一个非常方便的方法来生成初始化网络的state**
```python
initial_state = lstm_cell.zero_state(batch_size, dtype=)
#返回[batch_size, 2*len(cells)],或者[batch_size, s]
#这个函数只是用来生成初始化值的
```

**现在进行时间展开，有两种方法：**
**法一：**
使用现成的接口：
```python
tf.nn.dynamic_run(cell, inputs, sequence_length=None, initial_state=None,dtype=None,time_major=False)
#此函数会通过，inputs中的max_time将网络按时间展开
#cell:将上面的lstm_cell传入就可以
#inputs:[batch_size, max_time, size]如果time_major=Flase. [max_time, batch_size, size]如果time_major=True
#sequence_length:是一个list，如果你要输入三句话，且三句话的长度分别是5,10,25,那么sequence_length=[5,10,15]
#返回：（outputs, states）:output，[batch_size, max_time, num_units]如果time_major=False。 [max_time,batch_size,num_units]如果time_major=True。states:[batch_size, 2*len(cells)]或[batch_size,s]
#outputs输出的是最上面一层的输出，states保存的是最后一个时间输出的states
```
**法二**
```python
outputs = []
states = initial_states
with tf.variable_scope("RNN"):
	for time_step in range(max_time):
    	if time_step>0:tf.get_variable_scope().reuse_variables()#LSTM同一曾参数共享，
        (cell_out, state) = lstm_cell(inputs[:,time_step,:], state)
        outputs.append(cell_out)
```
**已经得到输出了，就可以计算loss了,根据你自己的训练目的确定loss函数**
##GRU
**GRU结构图**
来自[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
![图四](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png)
**tenforflow提供了tf.nn.rnn_cell.GRUCell()构建一个GRU单元**
```python
cell = tenforflow提供了tf.nn.rnn_cell.GRUCell(num_units, input_size=None, activation=tanh)
#参考lstm cell 使用
```