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



## 索引方式

```python
import numpy as np

val = np.array([1,2,3,4,5,6,7,8,9])
indiced_val = val[1:-1:3]  # [begin:end:step]
```



## 生成 grid

* np.ogrid

  ```python
  import numpy as np

  print(np.ogrid[1:4, 2:4])

  #[array([[1],
  #       [2],
  #       [3]]), array([[2, 3]])]
  ```

  可以看出，输出 `list of array` ，list 中的 第一个 array，是第一个参数的列向量形式，第二个不变



* np.mgrid

  ```python
  import numpy as np
  print(np.mgrid[1:4, 3:6])

  #[[[1 1 1]
  #  [2 2 2]
  #  [3 3 3]]
  #
  # [[3 4 5]
  #  [3 4 5]
  #  [3 4 5]]]
  ```

  生成一个 shape 为 `[2, height, width]` 的 `ndarray`，`height` 由第一个参数决定，`width` 由第二个参数决定。 第一个参数按列排，第二个参数按行排。

* np.meshgrid

