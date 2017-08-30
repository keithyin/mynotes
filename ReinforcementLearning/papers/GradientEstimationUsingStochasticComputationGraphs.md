# Gradient Estimation Using Stochastic Computation Graphs





**Key Contribution**

* assist researchers in developing intricate models involving a combination of stochastic and deterministic operations
* Enabling **Attention, Memory, control Actions**
* **Unbiased estimators for the gradient of the expected loss**
* **variance reduction techniques**






## 推公式

**都是对 $\theta$ 的导数哦**

* 第一个公式（x 是 stochastic）

$$
\begin{aligned}
\frac{\partial }{\partial\theta}\mathbb E[f(x)] &= \frac{\partial\sum_x p(x;\theta)f(x)}{\partial\theta} \\
&=f(x)\sum_x\frac{\partial p(x;\theta)}{\partial\theta}\\
&=f(x)p(x;\theta)\nabla_\theta\log p(x;\theta)\\
&=\mathbb E_x[f(x)\nabla_\theta\log p(x;\theta)]
\end{aligned}
$$

* 第二个公式（x是 deterministic，z 是 stochastic）（PD  path-wise derivative estimator）


$$
\frac{\partial}{\partial \theta}\mathbb E_z\biggr[f\Bigr(x(z,\theta)\Bigr)\Biggr] = \mathbb E_z\Biggr[\nabla_\theta f\Bigr(x(z,\theta)\bigr)\Biggr]
$$


* 第三个公式： $\theta$ 的阴魂不散之旅


$$
\begin{aligned}
\nabla_\theta\mathbb E_{z\sim p(\star,\theta)}\Biggr[f\Bigr(x(z,\theta)\Bigr)\Biggr] &= \nabla_\theta\sum_z p(z;\theta)f\Bigr(x(z,\theta)\Bigr)\\
&=\sum_z f\Bigr(x(z,\theta)\Bigr)\nabla_\theta p(z;\theta) + p(z;\theta)\nabla_\theta f\Bigr(x(z,\theta)\Bigr)\\
&=\mathbb E_{z\sim p(\star,\theta)}\Biggr[\nabla_\theta f\Bigr(x(z,\theta)\Bigr)+f\Bigr(x(z,\theta)\Bigr)\nabla_\theta\log p(z;\theta)\Biggr] 
\end{aligned}
$$




## Excerpt

> loss function is defined by an exception over a collection of random variables.

比如说监督学习的 loss function
$$
loss=\mathbb E_{x,y\sim\rho} \Biggr[\bigr(y-f(x)\Bigr)^2\Biggr]
$$
$x,y$ 就是随机变量啦，然后我们就是通过采样的方式来估计 loss function 的梯度的。对于一次 forward 过程来说，这个 $loss$ 是 deterministic 的，因为，毕竟 forward 时 $x,y$ 就已经确定了。



> optimizing loss functions that involves an expectation over **random variables** 。

这里倒是想起了一点，网络的输入也是随机的呀，感觉这些地方措辞有问题！！！！



> this estimator can be computed as the gradient of a certain differentiable function  (which  we call the **surrogate loss**).

论文中把 deterministic 和 stochastic 混合的问题，搞成了 deterministic + surrogate loss 问题，deterministic 就可以开心求导咯。

