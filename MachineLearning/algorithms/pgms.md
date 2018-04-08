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
$\Psi_C(X_C)$ 为 势函数（potential function）， 注意，势函数是定义在 最大团上的。一般情况下， 势函数是指数函数。



**线性链条条件随机场**
$$
P(y|x) = \frac{1}{Z(x)} \exp \Biggr(\sum_{i, k}\lambda_kt_k(y_{i-1}, y_i, x, i) + \sum_{i,l}\mu_ls_l(y_i, x, i)\Biggr)
$$
写成最大团乘积的形式：
$$
P(y|x) = \frac{1}{Z(x)} \prod_{C_i}\exp \Biggr(\sum_{k}\lambda_kt_k(y_{i-1}, y_i, x, i) + \sum_{l}\mu_ls_l(y_i, x, i)\Biggr)
$$
其中，$t_k(), s_l()$ 是特征函数， $\lambda_k, \mu_l$ 是对应的权值。

