# mxnet Symbol.Compose

```python
import mxnet as mx

data = mx.symbol.Variable('data')
net1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=10)
net1.list_arguments()
data2 = mx.symbol.Variable('data_2')
net2 = mx.symbol.FullyConnected(data=data2, name='fc2', num_hidden=10)
composed = net2(data_2=net1, name='composed') # Compose
print(composed.list_arguments())
# ['data', 'fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias']
# 从结果上来看，Compose 这个玩意可以换 Symbol 的 argument， 'data_2' 被net1 换掉了。
```

