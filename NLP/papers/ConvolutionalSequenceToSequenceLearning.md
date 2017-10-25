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

