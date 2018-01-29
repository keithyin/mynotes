# PCA&Whitening（主成分分析与白化）



## PCA

PCA（主成分分析）是一个降维算法，可以用来加速无监督特征学习算法。更重要的是，理解PCA可以帮助我们实现Whitening（白化）。白化对于很多算法来说是一个很重要的预处理步骤。



PCA可以允许我们通过一个低维的数据来近似的表示一个高维的数据。



PCA的主要思想是：将高维数据向高方差的方向映射。映射为低维数据。



**算法：**

假设我们有数据集 $\{x^1, x^2, x^3, x^4, ..., x^n\}$,那么PCA的算法如下：

1. 数据预处理：减均值，归一化
   1. $\mu=\frac{1}{n}\sum_{i=1}^nx^i$
   2. $x^i = x^i-\mu$
   3. $\sigma_j^2=\frac{1}{n}\sum_i(x_j^i)^2$
   4. $x_j^i=x_j^i/\sigma_j$
2. 计算 $\Sigma=\frac{1}{n}\sum_ix^ix^{(i)T}$
3. 计算$\Sigma$的特征值，特征向量，正交化之后，将特征值最高的几个对应的特征向量作为新的基底。



**如何将降维的数据recover成原数据：**

假设原始数据维度为 $k$ , 降维后的数据维度为 $r$ ,，协方差矩阵为 $U\in R^{k*k}$. $U$的每一列代表一个特征

$U^{-1} = U^T$

$x_{rot} = U^T*x_{origin}$

$x_{origin} = U*x_{rot}$ 当$x_{rot}$的维度为$r$时，应该填充为 $k$。





##  Whitening（白化）

We have used PCA to reduce the dimension of the data. There is a closely related preprocessing step called **whitening**，  If we are training on images, the raw input is redundant, since adjacent pixel values are highly correlated. The goal of whitening is to make the input less redundant。

![\begin{align}x_{{\rm PCAwhite},i} = \frac{x_{{\rm rot},i} }{\sqrt{\lambda_i}}.   \end{align}](http://ufldl.stanford.edu/wiki/images/math/e/2/9/e296118ba2bdf453dbe38426359f2230.png)

![\begin{align}x_{\rm ZCAwhite} = U x_{\rm PCAwhite}\end{align}](http://ufldl.stanford.edu/wiki/images/math/c/f/b/cfb1fa6b1049a5fdb2da4d7e88856751.png)

其中：第一个公式中的 $\lambda_i$表示第$i$个特征值。这就是两种白化方式。

