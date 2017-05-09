# python numpy

## numpy 常见使用错误
1.
```python
data.extend(para)
#参数不能为标量
```

2.
```python
a = [[1,2,3],[4,5,6]]
a = np.array(a)
# a的维度是(2,3)

a = [[1,2,3],[4,6]]
a = np.array(a)
#a的维度是(2,)
```
## numpy 扩维

```python
data_array = np.array([[1,2], [2,3]]) #shape[2, 2]
data_array[np.newaxis, :].shape # [1,2,2]
data_array[:,np.newaxis,:].shape #[2,1,2]
data_array[:, np.newaxis].shape #[2,1,2]
data_array[:,:,np.newaxis].shape #[2,2,1]
```

## repeat
```python
img = img[np.newaxis, :].repeat(3, axis=0)# axis 指定要在哪个维度上重复，
# 原img作为一个重复单元
```
