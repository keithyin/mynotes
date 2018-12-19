# Convolutional Sequence to Sequence Learning



## 各种符号表示

$k$ : 卷积核 kernel 的宽度，i.e. 一次看多少词

$m$ : 输入序列长度

$n$ : 输出序列长度

$f$ :  embedding size

$\mathbb x= (x_1, ... , x_m)$: 输入序列，id 形式

$\mathbb w = (w_1, ..., w_m) \in \mathbb R^{m*f}$ : 输入序列的 embedding 表示

$\mathbb p=(p_1, ..., p_m)\in \mathbb R^{m*f}$ : 输入序列的 绝对位置 embedding

$\mathbb e = (w_1+p_1, ..., w_m+p_m)$ : 表示终极版

$\mathbb z^l = (z^l_1, ..., z^l_m)$ : encoder 第 $l$ 层的输出

$\mathbb h^l = (h^l_1, ..., h^l_n)$ : decoder 第 $l$ 层的 输出



**Block**

* one dimensional convolution followed by a non-linearity


**问题：**

* 为什么会加速
* position embedding 有什么意义




**理解 position embedding**

* 位置信息有什么作用，

* 在 attention 中，一般关注的都是距离比较近的，position embedding 可以帮助处理这个问题

* > recurrent models typically do not use explicit position embeddings since they can learn where they are in the sequence through the **recurrent hidden state computation**. 

* CNN 解码和 RNN 解码不同的之处就在与 hidden state，RNN 有， CNN 没有，position embedding 是用来补充这部分信息的。



**补充实验**

* RNN + position embedding。
* RNN hidden state 到底保存了一些什么东西。

