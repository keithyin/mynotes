# 支持向量机



**不仅分的准，还要分的好！！！**

* **在正确分类的条件下， 离分界线最近的点的距离最大化**



* **二分类**
* `label` 为 $\{1,-1\}$



样本： $\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), (x^{(3)}, y^{(3)}),(x^{(4)}, y^{(4)}), ...\}$

**函数间隔 ：** $\hat \gamma^{(i)} = y^{(i)}(w\bullet x^{(i)}+b)$ 

**函数间隔最小值：** $\hat\gamma = \min_{i=1,2,3,...,N} \hat\gamma^{(i)}$

**几何间隔：** $\gamma^{(i)} = y^{(i)}(\frac{w}{||w||}\bullet x^{(i)}+\frac{b}{||w||})$

**几何间隔最小值：** $\gamma = \min_{i=1,2,3,...,N} \gamma^{(i)}$







## 线性可分支持向量机

**SVM** 想要最大分隔超平面， **所有点的几何间隔都要大于某个值**
$$
\begin{aligned}
&\max_{w,b} \gamma \\
&s.t. ~~y^{(i)}(\frac{w}{||w||}\bullet x^{(i)}+\frac{b}{||w||})\ge\gamma, i=1,2,3..,N.
\end{aligned}
$$
考虑到 函数间隔与几何间隔的关系式$\gamma = \frac{\hat \gamma}{||w||}$，此问题可以改写成：
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

## 为什么使用对偶形式

* 对偶问题往往更容易求解
  * 算法复杂度上的考虑，原问题的算法复杂度与 特征维度有关
  * 对偶问题的算法复杂度与 样本数量相关，SMO 算法。
* 可以引入核函数，推广到非线性分类问题上



## KKT 条件

**KKT条件有什么用？**

* 如果求得 $\alpha$ 的值，可以使用 kkt 条件计算得到 w 和 b 的值

$$
\begin{aligned}
 w^* - \sum_{i=1}^N\alpha^*_iy_ix_i &= 0 \\
 \sum_{i=1}^N \alpha^*_iy_i & = 0\\
 \alpha^*_i(y_i(w^*\bullet x_i + b^*)-1) &= 0\\
 ... \\
 ...
\end{aligned}
$$





## 测试的时候

$$
res = \text{sign } \Bigr(w^*\bullet x + b\Bigr)
$$



## Kernel Method





## SMO

上面提到，对偶问题的最终型式为：
$$
\max_{\alpha\ge0} \min_{w,b} L(w,b,\alpha)
$$
先求： $\max_{w,b} L(w,b,a)$ 。 然后得到的结果带入原表达式，会得到
$$
\begin{aligned}
&\min_\alpha &\frac{1}{2} \sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy^{(i)}y^{(j)}K(x^{(i)},x^{(j)}) - \sum_{i=1}^N\alpha_i \\
&s.t. & \sum_{i=1}^N\alpha_iy^{(i)} = 0 \\
&& 0\le\alpha_i\le C, ~~~i=1,2,3,...,N
\end{aligned}
$$
**SMO** 就是用来求以上 **凸二次规划** 问题的。 按照之前的套路的话，我们又会 拉格朗日一波流（因为上式也是一个带不等式约束的凸优化问题）， 但是 **SMO** 并不是这样，它采用启发式方法来求解, **只要 所有的参数都满足了 KKT 条件，那么这些参数必定是 最优参数无疑了**（这里提到的KKT条件是上面  $\min_\alpha$  问题的 KKT  条件。）。 **SMO** 在为这个目标一直努力着。



**那么，SMO 到底是怎么做的呢？**



由于 $\sum_{i=1}^N\alpha_iy^{(i)} = 0$ 的约束，SMO 一次无法只挑出一个 $\alpha_i$ 进行优化，因为只优化一个，固定其它，那么 $\alpha_i$ 必定是 **常数**。 所以 **SMO** 一次挑出来两个 $\alpha$  进行优化。



**如何优化**

* 挑第一个点：违反 KKT 条件最严重的点，由于某些等价关系，可以用 样本点的 数据来得到哪些违反的严重。
* 挑第二个点：



## 分析支持向量

**支持向量：** 

* 对于线性可分的来说，是离分类超平面最近的点。 i.e.  $w\bullet x +b =\pm1$ 的点。这时 $\alpha_i > 0$



## 遗留问题

* 支持向量的分析，用哪个公式分析最好，原问题的 $\min \max$ 公式？？？？
* SMO 如何挑选点的。
* ​