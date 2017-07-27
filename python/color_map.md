# color map

使用 `matplotlib` 或者 `matlab` ，在绘图的时候，经常会看到这个参数。那么这个参数到底是啥呢？



`color map` 实际上就是一个 三列的矩阵(或者说，shape 为 [N, 3]的 array )

* 矩阵中的值 取值范围 为 [0.,1.]
* 每一行代表一个颜色 (RGB)



## matplotlib

在使用 `python 的 matplotlib`库 时，可以使用现成的 `color map`，也可以自定义 `colormap` 。

```python
from matplotlib import colors
from matplotlib import cm
# 使用现成的 color map
cmap = cm.get_cmap('Set1')
res = cmap(score_map) # 会根据 score map 的值从 cmap 中找 color，返回的是 rgba 图像

# 自己定义
COLOR_MAP = ones(100, 3)
cmap = colors.ListedColormap(COLOR_MAP)
res = cmap(score_map)
```

