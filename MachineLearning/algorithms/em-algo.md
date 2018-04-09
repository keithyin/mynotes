# EM 算法

用于解决隐变量问题的算法

先看极大似然估计， 已收集到样本 $\{x^{(1)},x^{(2)},x^{(3)},x^{(4)},x^{(5)}, ..., \}$ $N$ 个
$$
\begin{aligned}
L(x^{(1)},x^{(2)},x^{(3)},x^{(4)},x^{(5)},... ;\theta) &=  \prod_{n=1}^N p(x^{(n)};\theta) \\
&= \prod_{n=1}^N \sum_zp(x^{(n)}, z;\theta) \\

\log L(...) &= \sum_n\log\sum_zp(x^{(n)}, z;\theta)
\end{aligned}
$$
这里为了将 `log-sum` 搞成 `sum-log` ， 使用了 `jason` 不等式
$$
\begin{aligned} 
\log L(...) &= \sum_n\log\sum_zp(x^{(n)}, z;\theta) \\
&= \sum_n\log\sum_z q_n(z)\frac{p(x^{(n)}, z;\theta)}{q_n(z)} \\
&\ge  \sum_n\sum_z q_n(z)\log\frac{p(x^{(n)}, z;\theta)}{q_n(z)} \\
\end{aligned}
$$
接下来就是，计算 $q_n(z)$ 的值，使其满足等式成立条件  (**E**)，然后再更新 $\theta$   (**M**) !!!

下面来推导 等式成立时 $q_n(z)$ 的值
$$
\begin{aligned}
\frac{p(x^{(n)}, z;\theta)}{q_n(z)} &= c \\
p(x^{(n)}, z;\theta) &= c*q_n(z) \\
\sum_z p(x^{(n)}, z;\theta) &= \sum_z c*q_n(z)\\
p(x^{(n)};\theta) = c \\
q_n(z) = \frac{p(x^{(n)}, z;\theta)}{c} \\
q_n(z) = \frac{p(x^{(n)}, z;\theta)}{p(x^{(n)})} \\
q_n(z) = p(z|x^{(n)};\theta)
\end{aligned}
$$
得到 $q_n(z)$ 的值之后，再使用梯度上升算法来更新 $\theta$

