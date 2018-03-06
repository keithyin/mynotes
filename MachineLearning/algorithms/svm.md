# 支持向量机

* **二分类**
*  `label` 为 $\{1,-1\}$



样本： $\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), (x^{(3)}, y^{(3)}),(x^{(4)}, y^{(4)}), ...\}$

**函数间隔 ：** $\hat \gamma^{(i)} = y^{(i)}(w\bullet x^{(i)}+b)$ 

**函数间隔最小值：** $\hat\gamma = \min_{i=1,2,3,...,N} \hat\gamma^{(i)}$

**几何间隔：** $\gamma^{(i)} = y^{(i)}(\frac{w}{||w||}\bullet x^{(i)}+\frac{b}{||w||})$

**几何间隔最小值：** $\gamma = \min_{i=1,2,3,...,N} \gamma^{(i)}$



## 线性可分支持向量机

**SVM** 想要最大分隔超平面
$$
\begin{aligned}
&\max_{w,b} \gamma \\
&s.t. ~~y^{(i)}(\frac{w}{||w||}\bullet x^{(i)}+\frac{b}{||w||})\ge\gamma, i=1,2,3..,N.
\end{aligned}
$$
考虑到 函数间隔与几何间隔的关系式，此问题可以改写成：
$$
\begin{aligned}
&\max_{w,b} \frac{\hat\gamma}{||w||} \\
&s.t. ~~y^{(i)}({w}\bullet x^{(i)}+{b})\ge\hat\gamma, i=1,2,3..,N.
\end{aligned}
$$
由于 $\hat\gamma$ 的取值对 **约束不等式** 没有任何影响，对**最优化问题**也没有任何影响，所以可以取 $\hat\gamma = 1$



所以最终的优化表达式变为：
$$
\begin{aligned}
&\min_{w,b} \frac{1}{2}||w||^2 \\
&s.t. ~~y^{(i)}({w}\bullet x^{(i)}+{b})\ge1, i=1,2,3..,N.
\end{aligned}
$$
**这是一个不等式约束最优化问题。**



**已知优化的表达式，构建 拉格朗日函数，为： **, 引入拉格朗日乘子 $\alpha_i\ge0$
$$
L(w,b,\alpha) = \frac{1}{2}||w||^2-\sum_i^N\alpha_iy^{(i)}(w\bullet x^{(i)}+b - 1)
$$
所以原始问题变成：
$$
\min_{w,b}\max_{\alpha\ge0} L(w,b,\alpha)
$$
原始问题的对偶问题为：
$$
\max_{\alpha\ge0} \min_{w,b} L(w,b,\alpha)
$$




