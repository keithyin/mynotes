# tensor transform



**`nd.pick(data=None, index=None, axis=_Null,  keepdims=_Null, out=None, name=None)`**

> 功能如其名，从 data 中 挑数据

* data : n-D NDArray
* index : (n-1)-D NDArray .  index 的shape 需要和除了 axis轴 之外的 data shape 一致！！！！ 
* axis : 沿着哪个轴挑， int 值 [0, n). 
* keepdims : 是否保持 维度不变。 如果为 False，则会降一维。如果为 True，axis 指定的轴为 1，其余不变

```python
from mxnet import nd

a = nd.array([[1, 2, 3], [4, 5, 6]])

print(nd.pick(a, index=nd.array([0,1,0]), axis=0, keepdims=True))
# [[ 1.  5.  3.]] axis=0 是 1, axis=1 是 3,没变

print(nd.pick(a, index=nd.array([0,1]), axis=1, keepdims=True))
# [[ 1.]
#  [ 5.]] axis = 1 是 1， axis=0 是 2， 没有变

# data 3-D 为例, index 为 2-D
# output[j,k] = data[index[j,k], j, k] . axis=0
# output[i,k] = data[i, index[i,k], k] . axis=1
# output[i,j] = data[i, j, index[i,j]] . axis=2
```

