# tensorflow做语音识别所需了解的API

## SparseTensor(indices, values, dense_shape)
- indices: 一个2D的 `int64 Tensor`,shape为`(N, ndims)`,指定了`sparse tensor`中的索引, 例如: indices=[[1,3], [2,4]]说明,`dense tensor`中对应索引为`[1,3], [2,4]`位置的元素的值不为0.

- values: 一个`1D tensor`,`shape`为`(N)`用来指定索引处的值. For example, given indices=[[1,3], [2,4]], the parameter values=[18, 3.6] specifies that element [1,3] of the sparse tensor has a value of 18, and element [2,4] of the tensor has a value of 3.6.

- dense_shape: 一个1D的`int64 tensor`,形状为`ndims`,指定`dense tensor`的形状.

相对应的有一个`tf.sparse_placeholder`,如果给这个`sparse_placeholder`喂数据呢?
```python
sp = tf.sparse_placeholder(tf.int32)

with tf.Session() as sess:
  #就这么喂就可以了
  feed_dict = {sp:(indices, values, dense_shape)}

```
> `tensorflow`中目前没有API提供denseTensor->SparseTensor转换

## tf.sparse_tensor_to_dense(sp_input, default_value=0, validate_indices=True, name=None)
把一个`SparseTensor`转化为`DenseTensor`.
- sp_input: 一个`SparceTensor`.

- default_value:没有指定索引的对应的默认值.默认为0.

- validate_indices: 布尔值.如果为`True`的话,将会检查`sp_input`的`indices`的`lexicographic order`和是否有重复.

- name: 返回tensor的名字前缀.可选.

## tf.edit_distance(hypothesis, truth, normalize=True, name='edit_distance')

计算序列之间的`Levenshtein` 距离

- hypothesis: `SparseTensor`,包含序列的假设.

- truth: `SparseTensor`, 包含真实序列.

- normalize: 布尔值,如果值`True`的话,求出来的`Levenshtein`距离除以真实序列的长度. 默认为`True`

- name: `operation` 的名字,可选.

返回值:
返回值是一个`R-1`维的`DenseTensor`.包含着每个`Sequence`的`Levenshtein` 距离.

`SparseTensor`所对应的`DenseTensor`是一个多维的`Tensor`,最后一维看作序列.
