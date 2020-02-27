# 创建tensor

```python
t = fluid.create_lod_tensor(np.ndarray([5, 30]), [[2, 3]], fluid.CPUPlace())
t = layers.fill_constant(shape=[1, log_scales.shape[1]], dtype="float32", value=-7.)
```





# tensor各种shape操作

```python
#concat & split
out2 = fluid.layers.concat(input=[x1,x2], axis=0)
x0, x1, x2 = fluid.layers.split(input, num_or_sections=[2, 3, 4], dim=2)


#rehape
data_2 = fluid.layers.fill_constant([2,25], "int32", 3)
dim = fluid.layers.fill_constant([1], "int32", 5)
reshaped_2 = fluid.layers.reshape(data_2, shape=[dim, 10])

# expand, the shape of expanded_1 is [2, 6, 2].
data_1 = fluid.layers.fill_constant(shape=[2, 3, 1], dtype='int32', value=0)
expanded_1 = fluid.layers.expand(data_1, expand_times=[1, 2, 2])

# expand_as, 有个 目标 shape 的tensor
data = fluid.data(name="data", shape=[-1,10], dtype='float64')
target_tensor = fluid.data(name="target_tensor", shape=[-1,20], dtype='float64')
result = fluid.layers.expand_as(x=data, target_tensor=target_tensor)
```

