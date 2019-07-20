# 概率图模型



## 无向图，马尔可夫网

**团与最大团**

> **无向图 G** 中，任意两个节点均有边连接的节点子集 称为为 团（clique）。
>
> 若 C 已经是 G 的一个团，并且不能添加任何一个节点使之成为更大的团，那么 C 为 最大团。



**概率图模型的因子分解**

> 基于最大团的因子分解

$$
P(X) = \frac{1}{Z}\prod_{X_C} \Psi_C(X_C)
$$

$Z$ : 规范化因子，保证概率为 1
$$
Z = \sum_X \prod_{X_C} \Psi_C(X_C)
$$
$\Psi_C(X_C)$ 为 势函数（potential function）， 注意，**势函数** 是定义在 最大团上的。一般情况下， **势函数**是**指数函数**。 势函数里面又包含了**特征函数**.



**可以看出, 无向图模型是使用 团上的势函数 定义概率**



**线性链条条件随机场**
$$
P(y|x) = \frac{1}{Z(x)} \exp \Biggr(\sum_{i, k}\lambda_kt_k(y_{i-1}, y_i, x, i) + \sum_{i,l}\mu_ls_l(y_i, x, i)\Biggr)
$$
* $i$ : 表示第  $i$ 个 最大团
* $k$ : 表示 第 $k$ 个特征, $l$ 也是一样

写成最大团乘积的形式：
$$
P(y|x) = \frac{1}{Z(x)} \prod_{C_i}\exp \Biggr(\sum_{k}\lambda_kt_k(y_{i-1}, y_i, x, i) + \sum_{l}\mu_ls_l(y_i, x, i)\Biggr)
$$
其中，$t_k(), s_l()$ 是特征函数， $\lambda_k, \mu_l$ 是对应的权值。



## 条件随机场的简化形式

$$
P(y|x) = \frac{1}{Z(x)} \exp \Biggr(\sum_{i, k}\lambda_kt_k(y_{i-1}, y_i, x, i) + \sum_{i,l}\mu_ls_l(y_i, x, i)\Biggr)
$$

由此式子可以看出, 条件随机场中的同一特征 $t_k(), s_l()$ 在各个最大团上都有定义, 所以可以对**同一个特征** 在各个位置求和, 将局部特征函数转化为一个全局特征函数.

设有 $K_1$ 个转移特征, $K_2$ 个状态特征, $K=K_1+K_2$


$$
f_k(y_{i-1}, y_i, x,i) = 
\begin{cases}
t_k(y_{i-1}, y_i, x, i), & k=1,2,3,...,K_1 \\
s_l(y_i, x,i), &k=K_1+l, l=1,2,3,..., K_2
\end{cases}
$$
这样将 **转移特征** 和  **状态特征** 搞在了一起表达.



然后对转移特征和状态特征在 各个位置 $i$ 求和
$$
f_k(y,x) = \sum_{i=1}^n f_k(y_{i-1}, y_i, x, i),  \\
k = 1,2,3,...,K
$$

* $k$ 表示第 $k$ 个特征
* $f_k(y,x)$ 现在表示全局的第 $k$ 个特征了.



现在再用 $w_k$ 表示特征 $f_k(y,x)$ ,
$$
w_k = 
\begin{cases}
\lambda_k, &k=1,2,3,...,K_1 \\
\mu_l, &k=K_1+l, l=1,2,3,...,K_2
\end{cases}
$$
现在, 条件随机场的表达式可以改写成
$$
P(y|x) = \frac{1}{Z(x)} \exp\sum_{k=1}^K w_kf_k(y,x)
$$
清爽多了.

再浓缩一下, 就可以写成下面这种形式, $K$ 为特征个数

$$
\begin{aligned}
w &= (w_1, w_2, ..., w_K)^T  \\
F(y,x)&=\Biggr(f_1(y,x), f_2(y,x), ..., f_K(y,x)\Bigr)^T
\end{aligned}
$$

$$
P_w(y|x) = \frac{1}{Z(x)} \exp w^TF(y,x)
$$



## 条件随机场的矩阵形式

引入起点和终点状态标记, $y_0=start, y_{n+1} = end$

对于观测序列的 $x$ 的每个位置 $i=1,2,3,..,n+1$ 定义一个 $m$ 阶矩阵, $m$ 是 $y_i$ 的取值个数
$$
\begin{aligned}
M_i(x) &= \Bigr[M_i(y_{i-1}, y_i|x)\Bigr]\\
M_i(y_{i-1}, y_i|x)&=\exp\Bigr(W_i(y_{i-1}, y_i|x)\Bigr)\\
W_i(y_{i-1}, y_i|x)&=\sum_{k=1}^Kw_kf_k(y_{i-1}, y_i, x, i)
\end{aligned}
$$
这时, 条件随机场又可以表示成, 这个是按照 最大团(时间步)表示
$$
P_w(y|x) = \frac{1}{Z(x)} \prod_{i=1}^{n+1}M_i(y_{i-1}, y_i|x)
$$



