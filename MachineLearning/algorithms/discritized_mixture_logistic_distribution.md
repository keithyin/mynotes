# Discritized Mixture of Logistic Distribution

**如何使用连续的分布建模离散的分布**

假设有 N 类, $[0,1,2,3,4,...,N-1]$,  $p(z)$ 为一个连续分布，高斯、logistic、etc.


$$
\mathbb P(x) = \int_{x-0.5}^{x+0.5} p(z) dz, ~~ x \in [1, 2, 3, ..., N-2]
$$
对于边界情况：
$$
\mathbb P(x=0) = \int_{-\infty}^{0+0.5} p(z)dz
$$

$$
\mathbb P(x=N-1)=\int_{N-1-0.5}^{\infty} p(z)dz
$$

**当 x 进行 scale 的时候，.5 这个直也会跟着 scale**

