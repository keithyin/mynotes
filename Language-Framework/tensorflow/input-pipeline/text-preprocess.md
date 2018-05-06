# 文本预处理

## 如何做 bucketing

```python
tf.contrib.data.group_by_window(
    key_func,
    reduce_func,
    window_size=None,
    window_size_func=None
)
"""
通过 key 来给 elements 分组, 将 window_size 个分为 一组 并 reduce 他们
key_func: 用来计算 key 的函数. 签名同 dataset 的输出
reduce_func: 如何处理同个 key 的数据
window_size: 可以理解成 batch_size
"""
```



```python
# tf.data.Dataset.padded_batch
padded_batch(
    batch_size,
    padded_shapes,
    padding_values=None
)
"""
将连续的dataset数据结合成一个 padded_batch

batch_size: batchsize

padded_shapes: A nested structure of tf.TensorShape or tf.int64 vector tensor-like objects representing the shape to which the respective component of each input element should be padded prior to batching. Any unknown dimensions (e.g. tf.Dimension(None) in a tf.TensorShape or -1 in a tensor-like object) will be padded to the maximum size of that dimension in each batch.

padding_values: (Optional.) A nested structure of scalar-shaped tf.Tensor, representing the padding values to use for the respective components. Defaults are 0 for numeric types and the empty string for string types.
"""
```

