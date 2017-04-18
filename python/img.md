# python 读取image
在python中我们有两个库可以处理图像文件，`scipy`和`matplotlib`.
## 安装库
```shell
pip install matplotlib pillow scipy
```
## 用法
```python
from scipy.misc import imread
data = imread(image_root)
#data是 ndarray对象
```

```python
import matplotlib.image as mpimg
data = mpimg.imread(image_root)
#data是 ndarray对象
```
