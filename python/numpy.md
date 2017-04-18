# python numpy 应该注意的地方
numpy 常见使用错误
**1**
```python
data.extend(para)
#参数不能为标量
```

**2**
```python
a = [[1,2,3],[4,5,6]]
a = np.array(a)
# a的维度是(2,3)

a = [[1,2,3],[4,6]]
a = np.array(a)
#a的维度是(2,)
```
