# Normalization
## local_response_normalization
local_response_normalization出现在论文"ImageNet Classification with deep Convolutional Neural Networks"中,论文中说,这种normalization对于泛化是有好处的.
$$ b_{x,y}^i = \frac{a_{x,y}^i}{ (k+\alpha\sum_{j=max(0,i-n/2)}^{min(0,i+n/2)}(a_{x,y}^j)^2)^\beta} $$
经过了一个conv2d或pooling后,我们获得了[batch_size, height, width, channels]这样一个tensor.现在,将channels称之为层,不考虑batch_size
- $i$代表第$i$层
- $a_{x,y}^i$就代表 第$i$层的 (x,y)位置所对应的值
- $n$个相邻feature maps.
- $k...\alpha ... n ...\beta$是hyper parameters
- 可以看出,这个函数的功能就是, $a_{x,y}^i$需要用他的相邻的map的同位置的值进行normalization
在alexnet中, $k=2, n=5, \alpha=10^{-4}, \beta=0.75$
```python
tf.nn.local_response_normalization(input, depth_radius=None, bias=None, alpha=None, beta=None, name=None)
'''
Local Response Normalization.
The 4-D input tensor is treated as a 3-D array of 1-D vectors (along the last dimension), and each vector is normalized independently. Within a given vector, each component is divided by the weighted, squared sum of inputs within depth_radius. In detail,
'''
"""
input: A Tensor. Must be one of the following types: float32, half. 4-D.
depth_radius: An optional int. Defaults to 5. 0-D. Half-width of the 1-D normalization window.
bias: An optional float. Defaults to 1. An offset (usually positive to avoid dividing by 0).
alpha: An optional float. Defaults to 1. A scale factor, usually positive.
beta: An optional float. Defaults to 0.5. An exponent.
name: A name for the operation (optional).
"""
```
- depth_radius:  就是公式里的$n/2$
- bias : 公式里的$k$
- input: 将conv2d或pooling 的输出输入就行了[batch_size, height, width, channels]
- return :[batch_size, height, width, channels], 正则化后
## batch_normalization
[论文地址](https://arxiv.org/pdf/1502.03167v3.pdf)
batch_normalization, 故名思意,就是以batch为单位进行normalization
- 输入:mini_batch: $In=\{x^1,x^2,..,x^m\}$
- $\gamma,\beta$,需要学习的参数,都是向量
- $\epsilon$: 一个常量
- 输出: $Out=\{y^1, y^2, ..., y^m\}$
算法如下:
(1)mini_batch mean:
$$\mu_{In} \leftarrow \frac{1}{m}\sum_{i=1}^m x_i$$
(2)mini_batch variance
$$ \sigma_{In}^2=\frac{1}{m}\sum_{i=1}^m(x^i-\mu_In)^2$$
(3)Normalize
$$\hat x^i=\frac{x^i-\mu_{In}}{\sqrt{\sigma_{In}^2 + \epsilon}}$$
(4)scale and shift
$$ y^i=\gamma\hat x^i + \beta$$
可以看出,batch_normalization之后,数据的维数没有任何变化,只是数值发生了变化
$Out$作为下一层的输入
函数:
tf.nn.batch_normalization()
```python
def batch_normalization(x,
                        mean,
                        variance,
                        offset,
                        scale,
                        variance_epsilon,
                        name=None):
```
Args:
- x: Input `Tensor` of arbitrary dimensionality.
- mean: A mean `Tensor`.
- variance: A variance `Tensor`.
- offset: An offset `Tensor`, often denoted $\beta$ in equations, or None. If present, will be added to the normalized tensor.
- scale: A scale `Tensor`, often denoted $\gamma$ in equations, or `None`. If present, the scale is applied to the normalized tensor.
- variance_epsilon: A small float number to avoid dividing by 0.
- name: A name for this operation (optional).
- Returns: the normalized, scaled, offset tensor.
对于卷积,x:[bathc,height,width,depth]
对于卷积,我们要feature map中共享 $\gamma_i$ 和 $\beta_i$ ,所以 $\gamma, \beta$的维度是[depth]
i
现在,我们需要一个函数 返回mean和variance, 看下面.
###tf.nn.moments()
```python
def moments(x, axes, shift=None, name=None, keep_dims=False):
# for simple batch normalization pass `axes=[0]` (batch only).
```
对于卷积的batch_normalization, x 为[batch_size, height, width, depth],axes=[0],就会输出(mean,variance), mean:[height,width,depth],variance:[height,width,depth]
