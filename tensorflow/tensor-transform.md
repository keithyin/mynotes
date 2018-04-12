# tf.scatter...



```python
tf.scatter_update(
    ref,
    indices,
    updates,
    use_locking=True,
    name=None
)
```

```python
 # Scalar indices
 ref[indices, ...] = updates[...]

 # Vector indices (for each i)
 ref[indices[i], ...] = updates[i, ...]

 # High rank indices (for each i, ..., j)
 ref[indices[i, ..., j], ...] = updates[i, ..., j, ...]
```





```python
tf.scatter_nd(
    indices,
    updates,
    shape,
    name=None
)
```

```shell
# 通过 shape 固定了输出的 shape
# updates 的 shape 为 indices.shape[:-1] + shape[indices.shape[-1]:]
# 可以看出，updates 的后几维是和 output_shape 一致的。
# indice决定了 ouput 的前面索引
# indice 的最后一个维度的 size 决定了 要在输出的哪个维度上填充值，值代表了在维度的哪个位置上填充值。
# 即：indice 的最后一维指定了 要在 res 哪个位置上更新。
res[*indice[i,j,..,z], ...] = updates[i,j,..,z,...]
len([i,j,..,z]) = indice.rank-2
```



