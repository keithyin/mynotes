# MaxEnt

* 为啥要 MaxEnt： 
  * 原因1： 熵增原理
  * 原因2：考虑 图像分类模型，假设一个 100*100 的 二值图像。图像的全集 $2^{100 * 100}$ ，可是我们可以采集到的样本却远远小于这个数字。假设我们用经验分布来统计得到 $\hat p(y|x)$ ， 在预测的时候，必定会出现一些没有出现过的 $x$, 导致 $\hat p(y|x) = 0$。这样的模型显然是不合理的。

**最大熵模型**

* 模型 $p(y|x)$
* 首先满足有限数量的限制
  * 这些限制都和期望值 相关
  * constraints on the average of some quantity you measure on the data
* 满足限制的条件下使得 $p(y|x)$ 熵最大



**Demo**

* 问题：预估等待cab的时间
* 样本：采到了一堆样本：4,3,2,5,6,3,2,4,5,6,8,3,4,5
* 模型: $P(x)$ 等待时间 x 的概率是多少
* contraint

$$
\mathbb E_{p_{model}(x)}(f(x)) = \mathbb E_{p_{experience}(x)}(f(x))
$$

* 其中 $f(x)=x$ 或者 $f(x)=x^2$ 或者 ......
* 



