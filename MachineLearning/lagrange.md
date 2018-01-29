# 拉格朗日乘子法

## 等式约束

拉格朗日乘子法是为了求解有约束问题而存在： 假设要求 函数 $f(x,y)$ 在等式约束条件 $\phi(x,y)=0$ 条件下的条件极值问题，我们一般这么构建 辅助函数：
$$
L(x,y,\lambda) = f(x,y)+\lambda \phi(x,y)
$$
仔细看一下这个式子：

* 当$x,y$ 满足约束条件时，得到的就是 $L(x,y,\lambda)=f(x,y)$



通过对这个辅助函数求梯度，我们会得到：
$$
\begin{cases}
f_x(x,y)+\lambda\phi_x(x,y)=0 \\
f_y(x,y)+\lambda\phi_y(x,y)=0\\
\phi(x,y) = 0
\end{cases}
$$
求解这个方程组，就可以得到约束条件下的最优解，仔细看一下，这个方程组是由两个部分构成的：

* 辅助函数 对 x 和 y 分别求偏导。
* 约束条件。





## 不等式约束

$$
\begin{aligned}
& \min_{x\in R^n} f(x) \\
& s.t. \\
&\begin{cases}
c_i(x)<=0, i=1,2,3,4,...,k \\
h_j(x) = 0, j=1,2,3,4,...l
 \end{cases}
\end{aligned}
$$

上面这个带有不等式约束条件的式子应该搞成什么样的非约束形式呢？如下所示
$$
L(x,\alpha,\beta) = \max_{\alpha\ge0,\beta} \Biggr[f(x)+\sum_{i=1}^k\alpha_ic_i(x)+\sum_{j=1}^l\beta_jh_j(x)\Biggr]
$$
观察上式：

* 如果条件不满足的话，$L(x,\alpha, \beta)=+\infty$
* 如果约束条件满足的话，$L(x,\alpha, \beta)=f(x)$
* 所以 $\min_xL(x,\alpha,\beta)$ 会更加喜欢 满足条件的 $x$

然后应该如何求的最优解呢？类比上面的等式约束，可以得到
$$
\begin{cases}
f'(x)+\sum_{i=1}^k\alpha_ic'_i(x)+\sum_{j=1}^l\beta_jh'_j(x) = 0 \\
c_i(x)<=0, i=1,2,3,4,...,k \\
h_j(x) = 0, j=1,2,3,4,...l
\end{cases}
$$
求解这三个式子来求得约束条件下的最优解。