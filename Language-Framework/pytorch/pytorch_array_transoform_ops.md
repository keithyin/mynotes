# pytorch Tensor变换 op 总结

维度： i.e. 几维，1-d，2-d 这样

shape：数据的形状



```python
torch.cat(seq, dim=0, out=None) -> Tensor
# 将 一系列的 tensor 给 通过给定的维度 连接起来
# Concatenates the given sequence of seq Tensors in the given dimension.
# seq ： a list of tensor 或者 a tuple of tensor
# dim： 指定把哪维连起来
# 注意：seq 中的 tensor： 维度必须一致，除了dim指定的维度，其它维度的shape必须保证一致
# 返回的 Tensor 的维度和seq 中 tensor 的维度一致
# 例子：
# 假设 seq = (a, b, c) a:shape[3,4], b:shape[4,4], c:shape[5,4]
# cat(seq, dim=0) 返回的 tensor shape 为 [12, 4]
```





```python
torch.chunk(tensor, chunks, dim=0) -> list of Tensor
# 将tensor 沿着给定的 dim 分割成 chunks 块
# Splits a tensor into a number of chunks along a given dimension.
# tensor：要被分割的 tensor
# chunks：int 值，指定要被分成几块
# dim：int 值，指定沿着哪个维度切块
```



```python
torch.gather(input, dim, index, out=None) -> Tensor
# Gathers values along an axis specified by dim.
# 通过 index 和 dim 从 input 中挑值，!! 挑的是标量
# index 的维度 与 input 的维度要一致
# 返回值的 shape 与 index 的 shape 一致
# ！！此接口应该是 通过遍历 index 来进行的赋值操作！！
# 官网中一个 3-D input 的例子
out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
```



```python
torch.index_select(input, dim, index, out=None) -> Tensor
# 通过 index 和 dim 从 input 挑值
# input：要被挑值的 tensor
# dim：int 指定维度
# index：1D LongTensor 包含要挑的索引
# ！！此接口应该是 通过遍历 index 来进行的赋值操作！！
out[i, :, :] = input[index[i], :, :]  # if dim == 0
out[:, i,: ] = input[:, index[i], :]  # if dim == 1
out[:, :, i] = input[:, :, index[i]]  # if dim == 2
```



```python
Tensor.masked_scatter_(mask, source)
# copies element from source into this tensor at positions where the mask is one
# mask : ByteTensor
# source : Tensor
```



```python
Tensor.scatter_(dim, index, src)
# writes ALL values from src into self at indices specified in the index
# index，self，src 的维度是一致的。
# ！！此接口应该是 通过遍历 index 来进行的赋值操作！！
out[index[i, j]][j] = src[i][j]  # if dim == 0
out[i][index[i, j]] = src[i][j]  # if dim == 1
```







## 对比

**`torch.gather` 与 `torch.index_select`**

* `torch.gather` 提供更加细粒度的操作



**`torch.gather` 与 `Tensor.scatter_`**

* `index` 参数 影响的 位置不同。