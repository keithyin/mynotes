# 机器学习算法---朴素贝叶斯

朴素贝叶斯属于机器学习中的 生成模型类别中，因为 朴素贝叶斯在学习 $P(X,Y)$ 联合概率分布。更具体的说，它是在学习 先验概率分布$P(Y=c_k)$和 条件概率分布$P(X=x|Y=c_k)$ .



为什么称之为朴素贝叶斯：

* 条件概率分布 $P(X=x|Y=c_k)$ 有指数级别的参数，直接估计是不可行的
* 所以朴素贝叶斯给了很强的假设，在已知类别的情况下，各特征是独立的。
* 即：$P(x_0,x_1,...,x_F|Y=c_k)=\prod_{i=0}^{F} P(x_i|Y=c_k)$  ,这样参数量就变成多项式级别的了。



朴素贝叶斯如何分类：
$$
P(Y=c_k|X=x) = \frac{P(Y=c_k)\prod_{i=0}^FP(x_i|Y=c_k)}{\sum_kP(Y=c_k)\prod_{i=0}^FP(x_i|Y=c_k)}
$$
就用上面这个式子找到最大的 $c_k$ 就可以了。



## 参数估计

可以看到，上面式子中，有很多值我们都不知道，如何估计它们的值呢？（就是参数估计）

**最大似然估计**

假设 $P(Y=c_k)= u_k, s.t. \sum_ku_k=1, u_k>=0$
$$
\begin{aligned}
likelihood &= \prod_k u_k^{N_k} \\
\log lh &= \sum_k N_k\log u_k\\
\frac{\partial \log lh}{\partial u_k} &= \frac{N_k}{u_k} 
\end{aligned}
$$
