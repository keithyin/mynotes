# rnn_cell
水平有限,如有错误,请指正!

本文主要介绍一下 `tensorflow.python.ops.rnn_cell` 中的一些类和函数,可以为我们编程所用

### run_cell.\_linear()
```python
def _linear(args, output_size, bias, bias_start=0.0, scope=None):

```
- args: list of tensor [batch_size, size]. 注意,list中的每个tensor的size 并不需要一定相同,但`batch_size`要保证一样.
- output_size : 一个整数
- bias: bool型, True表示 加bias,False表示不加
- return : [batch_size, output_size]
**注意: 这个函数的atgs 不能是 `_ref` 类型(tf_getvariable(), tf.Variables()返回的都是 `_ref`),但这个 `_ref`类型经过任何op之后,`_ref`就会消失**

**PS: `_ref` referente-typed is mutable**

### rnn_cell.BasicLSTMCell()
```python
class BasicLSTMCell(RNNCell):
  def __init__(self, num_units, forget_bias=1.0, input_size=None,
               state_is_tuple=True, activation=tanh):
"""
为什么被称为 Basic
It does not allow cell clipping, a projection layer, and does not
use peep-hole connections: it is the basic baseline.
"""
```
- num_units: lstm单元的output_size
- input_size: 这个参数没必要输入, 官方说马上也要禁用了
- state_is_tuple: True的话, (c_state,h_state)作为tuple返回
- activation: 激活函数
**注意: 在我们创建 cell=BasicLSTMCell(...) 的时候, 只是初始化了cell的一些基本参数值. 这时,是没有variable被创建的, variable在我们 cell(input, state)时才会被创建, 下面所有的类都是这样**

### rnn_cell.LSTMCell()
```python
class LSTMCell(RNNCell):
  def __init__(self, num_units, input_size=None,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=1, num_proj_shards=1,
               forget_bias=1.0, state_is_tuple=True,
               activation=tanh):           
```
- num_proj: python Innteger ,映射输出的size, 用了这个就不需要下面那个类了

### rnn_cell.OutputProjectionWrapper()
```python
class OutputProjectionWrapper(RNNCell):
  def __init__(self, cell, output_size):
```
- output_size: 要映射的 size
- return: 返回一个 带有 OutputProjection Layer的 cell(s)

### rnn_cell.InputProjectionWrapper():
```python
class InputProjectionWrapper(RNNCell):
  def __init__(self, cell, num_proj, input_size=None):
```
- 和上面差不多,一个输出映射,一个输入映射

### rnn_cell.DropoutWrapper()
```python
class DropoutWrapper(RNNCell):
  def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
               seed=None):
```
- dropout

### rnn_cell.EmbeddingWrapper():
```python
class EmbeddingWrapper(RNNCell):
  def __init__(self, cell, embedding_classes, embedding_size, initializer=None):
```
- 返回一个带有 embedding 的cell

### rnn_cell.MultiRNNCell():
```python
class MultiRNNCell(RNNCell):
  def __init__(self, cells, state_is_tuple=True):
```
- 用来增加 rnn 的层数
- cells : list of cell
- 返回一个多层的 cell
