# 集成模型



## 几个概念

**Blending**

* 已知 M 个模型，如何将这 M 个模型组合起来？
* 线性
* 非线性



**Bagging**

* 使用 **bootstrap** 从原本数据集中 重新采样获得新数据集。然后用新数据集训练。
* 使用采样的数据集训练，来获得 多样的 假设（模型）。



## Boosting 方法

[参考资料，机器学习技法，林 ](https://www.csie.ntu.edu.tw/~htlin/course/mltech18spring/doc/208_handout.pdf)

> 多个若分类器构成一个强分类器。 基函数的线性组合



* 所有弱分类器的训练都会用到数据集中的 **所有数据**
* 但是数据样本的权重不一样！！！



**使用 re-weighting 来获得 bootstrapping 的效果。** 

**最优 re-weighting**

假设，在 $t$ 时刻，我们训练好一个若分类器 $g_t(...)$ , 如何调节 $t+1$ 时刻样本的权重呢？
$$
\frac{\sum_{n=1}^Nu_n^{t+1}\mathbb I(y^{(n)}\ne g_t(x^{(n)}))}{\sum_{n=1}^N u_n^{t+1}} = \frac{1}{2}
$$
其中： $u_n^t$ : 表示，第 $n$ 个样本在 $t$ 时刻的权重。 $N$ 表示数据集样本数量。



简单点说，就是 

* 对于 $g_t$ 分类正确的样本 $u_n^{t+1} = u_n^t * \varepsilon_t$ 
* 对于 $g_t$ 分类错误的样本 $u_n^{t+1} = u_n^t * (1-\varepsilon_t)$
* 然后再归一化一下咯。

显然，像上面那么定义还是比较繁琐，所以， scaling factor 就来了：
$$
\text{scaling_factor} = \sqrt \frac{1-\varepsilon_t}{\varepsilon_t}
$$
所以，样本权重的变化策略变成了：
$$
\text{incorrect} \leftarrow \text{incorrect} \bullet \text{scaling_factor} \\
\text{correct} \leftarrow \text{correct} / \text{scaling_factor}
$$

* **然后再归一化**





样本的初始化权重为 $\frac{1}{N}$ 。



**每个时刻的优化目标为：**
$$
g_t \leftarrow \arg\min _{h\in \mathcal H} \Bigr(\sum_n u_n^t \mathbb I (y^{(n)} \ne h(x^{(n)}))\Bigr)
$$


**最终集成的模型为**
$$
f(x) = \sum_{m=1}^M \alpha_m g_m(x) \\
\alpha_m = \ln(\text{scale_factor}_m)
$$

* $m$ 和 $t$ 代表同一个意思。





## 提升树， boosting tree

上面提到 adaboost 方法是将多个 简单的若分类器 综合一下构成一个强分类器，但是没有提到具体的若分类器是什么。 当其中的若分类器为 **decision tree** 时，就是 **boosting tree** 。

**对于回归问题来说**
$$
f_M(x) = \sum_{m=1}^M T(x;\Theta_m)
$$

* 每一步的 decision tree 是在拟合的 残差。即： $c_{m,j}$ 来拟合当前的残差 $r_{m,i}$ ， 残差残差加起来，就是完整的值咯。



## GBDT

* 每一步 decision tree 在拟合 上一个的 树的输出的 loss 的 梯度。

**第 m 个 decision tree 要拟合的是目标是**
$$
r_{m,i} = -\Biggr[\frac{\partial L(y^{(i)}, f(x^{(i)})}{ f(x^{(i)})}\Biggr]_{f(x)=f_{m-1}(x)}
$$


**如何计算 $c_{m,j}$**
$$
c_{m,j} = \arg\min_c \sum_{x^{(i)}\in R_{m,j}} L(y^{(i)}, f_{m-1}(x^{(i)})+c)
$$


**如何更新 BDT**
$$
f_m(x) = f_{m-1}(x) + \sum_{j=1}^J c_{m,j}\mathbb I(x\in R_{m,j})
$$

$$
f(x) = \sum_{m=1}^M\sum_{j=1}^J c_{m,j} \mathbb I(x\in R_{m,j})
$$
**在这里回顾下 cart 如何做回归的吧**

* 对于每个特征
  * 对于每个特征值， 会将当前空间划分成两部分
    * 分别计算两部分 代表的值 $c_1, c_2$， 由 loss 函数来求得 最优 $c$ 值。 
    * 然后计算 loss。
* 挑 loss 最低的，对应着 特征 和 特征值，然后划分
* 重复上面操作。






## BDT 与 GBDT

**这里应该区分两个 loss**

* 决策树如何拟合的 loss。
* 模型的 loss。



**BDT** ： 每次使用 残差来划分区域，把残差相近的搞在一起。

* 认为残差相近的放在一起，整体 loss 就会降低。 这个是当然，值一样的放在一起，然后求个平均，loss 当然低。

**GBDT** ： 每次使用 梯度 划分 区域，梯度相近的放在一起。梯度相近意味着它们的值可能很相近。

* GBDT 中的决策树是用来拟合 梯度的。
* 为什么要拟合梯度 ??????????????????
* 难道 GBDT 认为，**梯度相似，代表着步长相似** ，假设两个样本的损失为 $(y^{(1)}-h(x^{(1)}))^2$ ，$(y^{(2)}-h(x^{(2)}))^2$ ，画出这两个 二次曲线，会发现，梯度相同的地方，它们到达最优点的步长是相同的！！！！
* 当 loss 为 平方损失函数的时候，GBDT == BDT。



**用 L1 loss 来看 GBDT**

*  GBDT 拟合梯度的方式的话：如果预测值比 target  大，就放在 `-1` 那堆， 如果预测值比 target 小，就放在 `1` 那堆。然后这两堆计算 $c_1, c_2$
* 如果使用残差拟合的话： 那他们估计要放到很多个堆里去咯。




## 遗留问题

* adaboost 的证明。
* gbdt 到底解决了什么问题，为什么会有那种问题。



## 参考资料

[https://www.csie.ntu.edu.tw/~htlin/course/mltech18spring/](https://www.csie.ntu.edu.tw/~htlin/course/mltech18spring/)

[https://www.quora.com/What-is-an-intuitive-explanation-of-Gradient-Boosting](https://www.quora.com/What-is-an-intuitive-explanation-of-Gradient-Boosting)