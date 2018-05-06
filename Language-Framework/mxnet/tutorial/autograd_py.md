# mxnet autograd 模块

**命令式编程自动求导模块**



## training 和 recording 状态

* `recording` :  当前是否在记录动态图。 负责判断当前是否需要 记录计算图。
* `training` : 当前是否是 训练 阶段。 



**training:**  不同阶段会影响 `Dropout` 和 `BatchNorm` 的行为。

```python
from mxnet import nd
from mxnet import autograd
from mxnet import gluon
import mxnet as mx

mx.random.seed(9)

val = nd.ones(shape=(10, 5))
w1 = nd.ones(shape=(5, 1))
b1 = nd.ones(shape=(1,))

print(type(b1))

w1.attach_grad()
b1.attach_grad()

with autograd.record():
    res = nd.dot(val, w1)
    with autograd.pause():
        c = res * 3  # just like tensor operation in pytorch.
    res2 = res + b1 + c
    # 在这儿 Dropout 是有效果的。
    final_res_train = nd.Dropout(res2, p=.5) 
    
# 在这个 位置 Dropout 是没有效果的。因为已经出了 training=True 范围。
final_res = nd.Dropout(res2, p=.5) 
 
print(final_res)
```



```python
autograd.record() # 默认情况下 recording 设置为 True， training设置为 True
autograd.pause() # 默认情况下 recording 设置为 False，training 设置为 False
autograd.train_mode() # recording 延用之前状态， training 设置为 True
autograd.predict_mode() # recording 延用之前状态，training 设置为 False
```







