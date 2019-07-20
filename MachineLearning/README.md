# 机器学习概览



## 基本概念

**生成模型 vs 判别模型** （模型的差别）

* 生成模型：$\mathbf P(x)$ 或者 $\{\mathbf P(x|y), \mathbf P(y)\}$  , 例子：隐马尔可夫模型，朴素贝叶斯
* 判别模型： $\mathbf P(y|x)$   ，逻辑斯特回归模型



**监督学习 vs 半监督学习 vs 无监督学习**  （数据的差别）

* 监督学习：样本都有 label
* 半监督学习：有些样本没有 label
* 无监督学习：样本没有 label



## 二分类问题

**精确率**

算法认为正类的样本中，有多少是正确的。
$$
P=\frac{TP}{TP+FP}
$$
**召回率**

算法挑出 正确的正例样本的比率。
$$
R=\frac{TP}{TP+FN}
$$
**准确率**
$$
Acc = \frac{TP+TN}{Num\_Samples}
$$
**F1**
$$
\frac{2}{F} = \frac{1}{P}+\frac{1}{R}
$$


## 朴素贝叶斯

生成模型，学习 $P(Y,X)$

**条件独立性假设**
$$
P(x_1,x_2,...,x_m|y) = \prod_i^m P(x_i|y)
$$
**模型的参数是**
$$
P(X_i=x_j|Y=y_k), P(Y=y_k)
$$
**模型训练**

* 使用 最大似然，或者 最大后验概率





## 决策树

> 树结构

* 如何构建决策树
* 如何剪枝

**样本**

$\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), ...\}$



**如何选择特征**

* 信息增益: 
* 信息增益比
* Gini 指数



**基本概念**

* 数据集的 熵： 即 label 作为随机变量的熵。和特征并无关系。
* ​



**ID3 算法**

> 使用信息增益来作为特征选择的标准，特征值的个数即是当前节点的 子节点个数。 一个特征只使用一次。
>
> 离散特征，离散 label

$$
\begin{aligned}
g(D,A) &= H(D) - H(D|A) \\
H(D|A) &= \sum_{i=1}^n\frac{|D_i|}{|D|} H(D_i)
\end{aligned}
$$

* $D(D)$ ：当前数据集的熵
* $n$ ：表示当前特征 $A$ 的取值个数
* $|D_i|$ ：表示 数据集 $D$ 中 , 特征 $A=A_i$ 的样本个数。

ID3 算法就是通过 信息增益来选择 但前要用哪个特征进行划分。



**C4.5 算法**

> 和ID3不同的是，C4.5 使用 信息增益比作为特征选择的依据
>
> 离散特征。离散 label

$$
\begin{aligned}
g_R(D,A) &= \frac{g(D,A)}{H_A(D)} \\
H_A(D) &= -\sum_{i=1}^n\frac{|D_i|}{|D|}\log(\frac{|D_i|}{|D|})
\end{aligned}
$$

$H_A(D)$ 表示 特征 $A$ 的熵。



**为什么信息增益率比信息增益要好**

使用信息增益，存在优先选择 **取值较多的特征** 的问题。取值较多的特征虽然能够很好的划分数据集，但是可能会导致 模型的复杂度增高。容易导致过拟合的问题。 其实信息增益率 也是一个控制模型复杂度的方法。



**如何剪枝**

* 确定 loss 函数，然后动态规划剪枝，保证剪枝后的 loss 小于当前 loss，且剪的越多越好。





**CART (classification and regression tree)**

> 分类 和 回归树， 不仅可以分类，而且可以回归。
>
> CART 假设决策树是 二叉树。
>
> 特征可连续，可离散，连续 label（回归嘛）。
>
> 对 分类树使用 Gini 指数， 对回归树使用 平方误差最小。



<font style="color:tomato">回归树：一个回归树对应着输入空间的一个划分以及在划分单元上的输出值。假设已经将输入空间划分成 $M$ 个单元，且每个单元 $R_m$ 上都有一个输出值 $c_m$ ，则回归树模型可以表示为</font>
$$
f(x) = \sum_{m=1}^M c_mI(x\in R_m)
$$
如何对输入空间进行划分，采用启发式的方法，选择第 $j$ 个变量 $x_j$ 和它取的值 $s$ 作为切分变量和切分点，并将其划分成两个区域： $R_1(j,s) = \{x|x_j\le s\}$ 和 $R_1(j,s) = \{x|x_j\gt s\}$ 。然后求解：
$$
\min_{j,s} \Biggr[\min_{c_1}\sum_{x^{(i)}\in R_1(j,s)}(y_i-c_1)^2 + \min_{c_2}\sum_{x^{(i)}\in R_2(j,s)}(y_i-c_2)^2 \Biggr]
$$

* 遍历 $j$
  * 扫描切分点 $s$， 如果离散的话，所有特征值遍历一遍，如果连续的话，先量化，然后遍历一遍。
    * 求 $c_1$ 和 $c_2$ 



<font style="color:tomato"> 分类树：使用 Gini index 选择最优特征，同时决定该特征的值为 最优二值切分点。 </font>

> 特征离散， label 离散。

**基尼指数：** 分类问题中，假设有 $K$ 个类，样本点属于第 $k$ 类的概率为 $p_k$ ，则概率分布的基尼指数定义为： 
$$
\text{Gini}(p) = \sum_{k=1}^K p_k(1-p_k) = 1-\sum_{k=1}^Kp_k^2
$$
对于给定的样本集合，Gini-index 为：Gini 指数越小，说明选择的特征越好。
$$
\text{Gini}(D) = 1-\sum_{k=1}^K\Bigr(\frac{|D_k|}{|D|}\Bigr)
$$

* $|D_k|$ : 数据集中，第  $k$  类的样本个数。



在特征 $A$ 下，集合 $D$ 的基尼指数定义为
$$
\text{Gini}(D,A) = \frac{|D_1|}{|D|}\text{Gini}(D_1) + \frac{|D_2|}{|D|}\text{Gini}(D_2)
$$


## Logistic Regression

> 二分类问题

$$
\begin{aligned}
P(Y=1|x) &= \frac{\exp(wx+b)}{1+\exp(wx+b)} \\
P(Y=0|x) &=  \frac{1}{1+\exp(wx+b)}
\end{aligned}
$$

然后，最大似然估计法计算 模型参数 $w, b$



## Maximum Entropy Model （最大熵模型）

**最大熵原理：** 学习概率模型时，在所有可能的概率模型（分布）中，熵最大的模型是最好的模型。通常约束条件确定了概率模型的集合，使用最大熵原理来从满足的约束条件的概率模型集合中挑选一个最好的概率模型（分布）。


$$
\begin{aligned}
H(P) &= -\sum_xP(x)\log P(x) \\
0&\le H(P) \le \log(|X|)
\end{aligned}
$$

* $|X|$ : X的取值个数。当 $|X|$ 服从均匀分布时，熵最大。



**最大熵模型：** 最大熵原理是统计学习的一般原理，将它应用到分类得到最大熵模型。

模型： $P(Y|X)$, 学习的目标是给定数据集，使用最大熵原理选择最好的分类模型。







## 偏差与方差

[http://scott.fortmann-roe.com/docs/BiasVariance.html](http://scott.fortmann-roe.com/docs/BiasVariance.html)



**偏差**

*  difference between the **expected (or average) prediction of our model** and the correct value which we are trying to predict.
* 模型预测的平均值 与 真实值之间的差距
* 模型为啥会有平均值： 同一个模型，重复训练多次，会得到多组参数，每组参数看作一个模型
* 另一种看法： 训练集的准确率



**方差**

* variability of a model prediction for a given data point
* 一个模型训练多次，看它对同一个点的预测的 变化程度。
* 另一种看法：dev set 的准确率



**降低偏差：**

* 增加模型复杂度
* 换用更好的模型



**降低方差：**

* 增加数据量
* 降低模型复杂度
* ​

