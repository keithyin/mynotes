# pytorch 卷积操作总结
本文主要是介绍一下`pytorch`中关于卷积操作的`API`，和关于其的一点总结。

`pytorch`中关于卷积的运算一共有6个：

* conv1d
* conv2d
* conv3d
* conv_transpose1d
* conv_transpose2d
* conv_transpose3d

分别对应正向卷积三个和反卷积三个。这里主要介绍`torch.nn.functionals`中的接口，因为`torch.nn`中定义的接口也就是把定义`weight`和调用函数都帮你做了而已。

## 正向卷积

* `torch.nn.functional.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)`
* `torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)`
* `torch.nn.functional.conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)`

可以看出，三个对应不同维度的卷积函数，参数是一样的。

- input：代表输入数据`Tensor`
  - 一维时，`shape` 为 `(minibatch, in_channels, iw)`, `iw` 即 `input width`。
  - 二维时，`shape` 为 `(minibatch, in_channels, ih， iw)`， `ih` 即 `input height`。
  - 三维时，`shape` 为 `(minibatch, in_channels, it， ih， iw)`， `it` 即 `input time`。

- weight： 表示卷积核参数
  - 一维时，`shape` 为 `(out_channels, in_channels/groups, hw)`, `kw` 即 `kernel width`。
  - 二维时，`shape` 为 `(out_channels, in_channels/groups, kh， ik)`， `kh` 即 `kernel height`。
  - 三维时，`shape` 为 `(out_channels, in_channels/groups, kt， kh， kw)`， `kt` 即 `kernel time`。

- bias: 表示bias参数，如果卷积之后需要加个bias的话，可以传入
  - 无论几维卷积，shape都是 `(out_channels)` 。

- stride： 用来表示步长的参数。从这个参数上可以看出pytorch的参数设计上，能用一个数，绝不用两个数
  - 一维时： 是一个 `int` , 用来表示在输入数据宽度上的步长
  - 二维时： `int` 或者 二元`tuple`。如果是 `int`，那么高和宽维度上的步长都为这个值。如果是`tuple`，高和宽的步长需要和`tuple`的对应元素对应。
  - 三维时： `int` 或者 三元`tuple (st*sh*sw)`。

- padding：设置填充多少0的参数，默认情况下是不填充。参数的设计和stride有点类似
  - 一维时： `int` ，用来表示在宽的左右两边分别扩展的像素数。
  - 二维时： `int`或 二元 `tuple` ，表示对于 高 和 宽 两边分别扩展的像素数。
  - 三维是： `int`或 三元 `tuple`， 表示对于 时间、高 和宽两边的分别扩展的像素。

- dilation: 这个用默认值就行了，感觉目前没啥用。

- groups： 这个是将`input channels` 分组。通常情况下，我们是将`input channels`所有看作一组。

- 输出：
  - 一维：`shape` 为 `(minibatch, out_channels, out_width)`
  - 二维：`shape` 为 `(minibatch, out_channels, out_height, out_width)`
  - 三维：`shape` 为 `(minibatch, out_channels, out_time, out_height, out_width)`

## 反卷积/解卷积

* `torch.nn.functional.conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1)`
* `torch.nn.functional.conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1)`
* `torch.nn.functional.conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1)`

思考卷积的转置这些函数，先从正向的考虑，然后思考这些参数的意思，还是比较清楚的。
现在**假设**我们有一个 正向的 卷积过程
```python
res = conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
```
`res`就是正向卷积的结果，现在我们基于这个正向卷积解释 `conv_transpose` 参数的意义

- input
  - 就是`res`
- weight
  - 就是正向的`weight`，直接填入正向的卷积核。shape为`(in_channels, out_channels/groups, hw)`,和正向卷积核的shape的解释对比，发现，正向的`in_channels`，就是反向的`out_channels`,很合理。
- bias
  - 加到解卷积输出上的`bias`。
- stride
  - 正向的`stride`
- padding
  - 正向进行的`padding` 操作。这个是移除前向过程的 pad 。
- output_padding
  - 这个是对反卷之后的 `shape` 进行修补， 为什么要修补呢？ stride为1的时候是不需要修补的，因为正向卷积的时候，不会浪费输入的值。但是如果stride=4，那就可能会有浪费的值了。为什么呢？假设输入 `width=10`，如果进行`stride=4 kernel=6`卷积后返回的值的`width=2`.那么`width=11`呢，返回的`shape=2`，`width=12`呢，返回还是`width=2`。那么只有`width=2`的结果，如何获得`width=10/11/12`输入呢？所以需要 `output_padding` 进行补偿。
- groups
  - 正向的`group`
- dilation
  - 正向的`dilation`


## 卷积后的feature map的大小的计算

**先考虑 dilation=1 的情况， 就是我们最常用的情况**

`Lout=floor((Lin+2∗padding−dilation∗(kernel_size−1)−1)/stride+1)`

$$
L_{out} = floor\Biggr(\frac{L_{in}+2∗padding−kernel\_size}{stride}+1\Biggr)
$$

其中： 

* $L_{in}+2*padding$ 是 pad后的 输入 featuremap 的大小。
* 用 `floor` 的原因是， 如果剩余的部分不够卷积运算卷的，就忽略

**再看看有 dilation 的情况：**
如果有了 dilation后， 那么 kernel_size 就变成了 $$kernel\_size=dilation∗(kernel\_size−1)+1$$

所以，通用的公式是
$$
L_{out} = floor\Biggr(\frac{L_{in}+2∗padding−\Bigr(dilation∗(kernel\_size−1)+1\Bigr)}{stride}+1\Biggr)
$$


反卷积的计算就是上面公式已知 `Lout` 求 `Lin`

`Wout=(Win−1)∗stride[1]−2∗padding[1]+kernel_size[1]+output_padding[1]` 。对 `outpadding` 并没有乘



反卷积计算的时候，应该是:

* 将输入 膨胀填充成 `stride*(Lin-1)+1 +2*(ks-1)` 长或宽。 `stride*(Lin-1)+1` 是膨胀后的大小（类似于 dilation 后的 核）
* 然后用 卷积 运算把上面填充后的 卷一遍, 以 **`stride=1`** 的形式。
* 这时候能得到一个 `stride*(Lin-1)+ks` 长或宽 的 feature map。 
* 然后再适当的 `crop` 或 `output_padding`。

