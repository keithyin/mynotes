# EM 算法

> 用于含有**隐变量**的概率模型 **参数的极大似然估计**



## 观测变量与隐变量

可以看到的变量是 观测变量，看不到的变量是隐变量。



**三硬币模型：**

有三个 硬币 A,B,C, 正面出现的概率分别是 $\pi, p, q$，实验这么设计：先掷硬币 A，根据A 的结果再掷硬币 B或C，如果A为正，掷B，否则，掷C，然后掷出来的结果，如果为正，就是 1, 否则，就是 0. 



数学表达式为：
$$
\begin{aligned}
P(y|\theta) &= \sum_zP(y,z|\theta) \\
&=\sum_zP(z|\theta)P(y|z,\theta)\\
&=\pi p^y(1-p)^{1-y} + (1-\pi)q^y(1-q)^{1-y}
\end{aligned}
$$


于是乎：

* 观测变量： B或C掷出来的结果
* 隐变量：A掷出的结果
* 模型参数： $\theta=\{\pi, p, q\}$



## 最大 对数似然

由上部分可知：
$$
P(y|\theta) ==\pi p^y(1-p)^{1-y} + (1-\pi)q^y(1-q)^{1-y}
$$
然后考虑最大似然：
$$
P(\mathbf Y|\theta) = \prod_n \pi p^{y^n}(1-p)^{1-y^n} + (1-\pi)q^{y^n}(1-q)^{1-y^n}
$$
最大化 log似然
$$
\max_\theta  \sum_n \log\Biggr[p^{y^n}(1-p)^{1-y^n} + (1-\pi)q^{y^n}(1-q)^{1-y^n}\Biggr]
$$
这个问题没有解析解只能通过迭代的方式求解，这时候，EM 算法就出马了



## EM算法

$$
L(\theta) = \log P(Y|\theta) = \log \sum_zP(Y,Z|\theta)
$$

没有解析解的问题出在 $\log \sum_z P(Z|\theta)P(Y|Z,\theta)$ ，和的 log 上。


$$
\begin{aligned}
L(\theta) &= \log P(Y|\theta) \\
& = \log \sum_zP(Y,Z=z|\theta) \\
&= \log \sum_z Q(z) \frac{P(Y,Z=z|\theta)}{Q(z)} \\
&\ge \sum_z Q(z) \log \frac{P(Y,Z=z|\theta)}{Q(z)} 
\end{aligned}
$$
等号相等的条件是：
$$
\frac{P(Y,Z=z|\theta)}{Q(z)} = c
$$
由于 $\sum_zQ(z) = 1$ 

所以：
$$
\begin{aligned}
P(Y,Z=z|\theta) &= c Q(z) \\
\sum_z P(Y,Z=z|\theta) &= c\\
P(Y|\theta) &= c \\
Q(z) &= \frac{P(Y,Z=z|\theta)}{Q(Y|\theta)} \\
Q(z) &= P(z|Y,\theta)
\end{aligned}
$$
所以，EM 算法的精髓就是，先使等号成立，然后再最大化不等式右边。



EM算法基本步骤：

* E-step : for each i : $Q_i(z^{(i)}) = p(z^{(i)}|x^{(i)},\theta)$
* M-step: $\theta := \arg_\theta \max \sum_i\sum_{z^{(i)}}Q_i(z^{(i)})\log\frac{p(x^{(i)},z^{(i)}|\theta)}{Q_i(z^{(i)})}$ , loop 直到收敛。

