# torch.sparse

`torch.sparse.FloatTensor`

```python
class torch.sparse.FloatTensor:
    def __init__(self, indices, values, size):
        pass
```

* `indices` ： 一个二维的 `torch.LongTensor`，`shape=[sparse_dim, num_none_zero]`
  * `sparse_dim` : 稀疏 `tensor` 的维度
  * `num_none_zero` ： 非0值个数
* `values`：一个二维或一维的 `tensor` , `shape=[num_none_zero, none_sparse_dim]`
  * `non_sparse_dim` : 非稀疏部分的维度（一个 tensor，可能在某维度往上才会展示出稀疏的特征。）
* `size`：稀疏矩阵的`shape`， `torch.Size` 对象



```python
# 当 sparse tensor 只是对 idx 和 values 进行了引用。牵一发，都会改变
idx = torch.LongTensor([[0, 1, 1],
                        [2, 2, 0]])
values = torch.FloatTensor([10, 20, 30])
sp_tensor = sparse.FloatTensor(idx, values, torch.Size([2, 3]))
print(sp_tensor)
idx[0, 0] = 1
print(sp_tensor)
values[1] = 80
print(sp_tensor)
```



## 参考资料

[https://pytorch.org/docs/stable/sparse.html](https://pytorch.org/docs/stable/sparse.html)

