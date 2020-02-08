# variational auto encoder

[论文地址](http://cn.arxiv.org/pdf/1312.6114.pdf)

[一个很好的博客地址](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)

[又一篇博客](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)



# Variational Inference

**Probabilistic Pipeline**

- make assumptions
- discover patterns
- predict & explore



**inference**

- **uncover the hidden variables given the observed data**
- what does this model say about this data
- 目的是寻找一个 general 和 scalable 的方法来做 inference



**VI**

* turns inference into optimization problem.
* 假设 hidden variable 符合一个简单的概率分布$q(\mathbb z; \mathbb \lambda)$，vi 不停优化这个分布的参数 $\mathbb \lambda$ 使其越来越接近真实的后验概率分布。

**complete conditional**

* **conditional of a latent variable**  given the **observations** and **other latent variables**
* Assume each complete conditional is in the exponential family.


$$
\text{ELBO} = \mathbb E_{q(\mathbf z;\mathbf \lambda)}\Biggr[\log p(x,z)-\log q(z;\lambda)\Biggr]
$$




**SVI**

* 先对 ELBO 积分，得到一个表达式，再求导
* Stochasitic 表示的是 样本采样的 随机性，mini-batch



**BBVI**

* 先对 ELBO 求导， 然后再用采样的方法求积分
* 这里有两个随机性，
  * 样本的随机性
  * 求积分时采样的 隐变量的 随机性

$$
\nabla_\lambda\text{ELBO} = \mathbb E_{q(\mathbf z;\mathbf \lambda)}\Biggr[\nabla_\lambda\log q(z;\lambda)\Bigr[\log p(x,z)-\log q(z;\lambda)\Bigr]\Biggr]
$$

**Score Function Estimator**

* 上面的式子就叫做 score function estimator， 或者 likelihood ratio 或者 REINFORCE  gradient。
* $\mathbf z^i \sim q(\mathbf z; \lambda)$ , **采样计算梯度**
* 这个算法基本不 work， 因为梯度估计的方差太大。



**Control Variates**

* 目的：$x \sim Dist(\mu, \sigma)$ , 无论 x 采到均值点，还是离均值点较远，$f(x)$ 都要接近于均值！！！及，找到一个 具有小方差的 $\hat f(x)$ 但是和 $f(x)$ 同均值的函数来替代 $f(x)$ 。

$$
\hat f(z) = f(z) - a\Bigr(h(z)-\mathbb E[h(z)]\Bigr)
$$

* 对于 BBVI 来说， $h(z)$ 可以选择 $\nabla_\lambda \log q(z; \lambda)$ 



**Path-Wise Estimator**

* e.g. re-parameterization trick



**Amortized Inference**

* 不为每个 $\mathbf x^i$ 都预测出来一个 $\mathbf z^i$, **训练一个 $\mathbf z^i = f(\mathbf x^i)$ 的函数**。
* 这样，如果一个新的 样本进来，就可以计算出其 隐变量表示
* 之前就不行。

