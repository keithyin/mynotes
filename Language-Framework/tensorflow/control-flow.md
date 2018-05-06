# Control Flow
介绍几个 `tf` 中的 `control flow` `API`，什么是`control flow`呢？就是 `if-else`, `while`, `case`这些。
## tf.cond(pred, fn1, fn2, name=None)
等价于三目运算符
```python
res = fn1() if pred else fn2()
```
需要注意的是：
`fn1`, `fn2` 为 `callable`， 一般用 `lambda`（匿名函数）定义。
`pred` 是个 标量
例子：
```python
z = tf.mul(a, b)
result = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))
```

## tf.case(pred_fn_pairs, default, exclusive=False, name='case')

## 参考资料
[https://www.tensorflow.org/versions/r0.12/api_docs/python/control_flow_ops/control_flow_operations#cond](https://www.tensorflow.org/versions/r0.12/api_docs/python/control_flow_ops/control_flow_operations#cond)
