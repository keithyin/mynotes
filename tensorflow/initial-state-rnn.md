# 如何初始化LSTM的state
`LSTM` 需要 `initial state`。一般情况下，我们都会使用 `lstm_cell.zero_state()`来获取 `initial state`。但有些时候，我们想要给 `lstm_cell` 的 `initial state` 赋予我们想要的值，而不是简单的用 `0` 来初始化，那么，应该怎么做呢？
当然，当我们设置了`state_is_tuple=False`的时候，是很简单的，当`state_is_tuple=True`的时候，应该怎么做呢？
## LSTMStateTuple(c ,h)
可以把 `LSTMStateTuple()` 看做一个`op`
```python
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import LSTMStateTuple

...
c_state = ...
h_state = ...
# c_state , h_state 都为Tensor
state_tuple = LSTMStateTuple(c_state, h_state)

```

当然，`GRU`就没有这么麻烦了，因为`GRU`没有两个`state`。
