# Casual Inference & Uplift Modeling

> uplift modeling 是解决 casual inference 的一个建模方式

因果推理建模中的两个重要概念：1）potential outcomes，2）causal effects

问题的框架如下：

* $N$ individuals indexed by $i$
* $Y_i(1)$ 表示 第 $i$ 个人接受 active treatment 的outcome, $Y_i(0)$ 表示第 $i$ 个人接受 control treatment 的 outcome。
* causal effect $\tau_i = Y_i(1) - Y_i(0)$

用 $X$ 表示特征，$x_i$ 表示 individuals 的 特征向量，则 conditional average treatment effect(CATE) 可以写成
$$
\tau(x_i) = \mathbb E[Y_i(1) - Y_i(0) | X = x_i]
$$

* 相比于 $\tau_i$ 来说，researcher 更加关注 $\tau(x_i)$ ，因为用了特征之后，模型具有更强的泛化性了。也就是具有了预测的能力。
  * CATE就是 uplift modeling 建模的东西？
* **这里需要注意的一点是**：有时候，我们会将 `CATE` 建模为 $\mathbb E[Y_i(1)|X_i=x, W_i=1] - \mathbb E[Y_i(0)|X_i=x, W_i=0]$ . 但是这个公式并不总是正确的。
  * 该式子正确的前提是：已知$X_i$ 情况下，$W_i$ 与  $Y_i(1)$ 和 $Y_i(0)$ 是独立的。$\\{Y_i(1), Y_i(0) \\}\perp\!\!\!\perp W_i | X_i$ 
    * 独立：一个事件的发生，不影响另一个事件的发生。
    * 这里并不要求：$P(W_i=0) = \frac{1}{2}$



# uplift

uplift caused by taking the action conditional on  $X_1, X_2, ..., X_m$
$$
uplift = P(Y=1|X_1, ..., X_m, G=T) - P(Y=1|X_1, ..., X_m, G=C) \\\\
uplift = P^T(Y=1|X_1, ..., X_m) - P^C(Y=1|X_1, ..., X_m)
$$




# Two Models

* 使用两个独立的概率模型：一个建立在 control 数据集上，一个建立在 treatment 数据集上。同一个特征输入到这两个模型得到的结果相减就能得到 uplift
* 广义two models模型：一旦输入特征加上了 control/treatment group的标记，那就是 two models模型了。



# Class Transform

参考论文 Uplift modeling for clinical trial data

传统分类建模：

* 根据输入特征预测类别的概率

Uplift 建模：

* predict the difference between class probabilities in the treatment group and the control group.

Uplift 建模方式适合：

* 临床试验
* 营销



对于临床试验来说，目的是看 treatment group 和 control group 有什么 diff：

* 如果使用传统分类建模：数据集的构建就全是treatment group的数据，然后在该数据集上构建一个模型，然后看受treatment之后，是否会有效果的概率。
* 如果使用uplift建模：数据集的构建就会包含treatment group & control group，模型会显式的建模两个group的输出概率的diff。使用uplift建模，最终得到的模型还可以得到 treatment 对哪些病人更加有效。



TwoModels：

* 优点：
  * 任何分类模型都可用
  * 如果uplift与class强相关，或者训练数据足够多使得模型训练的很准确，那么 two models 模型表现的也会挺好。
* 缺点：当uplift和class没啥关系，模型就会专注于预测class，而非预测 `weaker uplift signal` 。详细解释见 https://stochasticsolutions.com/pdf/sig-based-up-trees.pdf



* 分类模型：输入 $X_1, ..., X_m$，输出 $[0, 1]$ 来表示 $Y=1$ 的概率
* class transform uplift 模型：输入 $X_1, ..., X_m$，输出 $[-1, 1]$ 来表示 $P^T(Y=1)-P^C(Y=1)$



Class Transformation


$$
Z=
\begin{cases}
1, (T=1,Y=1) or (T=0, Y=0)\\\\
0, else
\end{cases}
$$
当 $Z$ 表示该含义时，我们可以得到且 $P(G=T|X_1, ..., X_m) = \frac{1}{2}$ 时，我们可以推出以下等式
$$
P^T(Y=1|X_1, ..., X_m) - P^C(Y=1|X_1, ..., X_m) = 2P(Z=1|X_1, ..., X_m) - 1
$$
模型只需要根据训练集预测出 $P(Z=1|X_1, ..., X_m)$ ，就可以根据上述变换得到 $uplift$

> 如果 $P(G=T|X_1, ..., X_m) != \frac{1}{2}$ 咋整呢？this is a problem。



# Modeling Uplift Directly

* logistic regression
* k-nearest neighbors
* modified svm
* Tree-based
* Ensemble methods



# Evaluation

> 我们无法看到 treated/not treated在同一个人身上的影响，所以我们没有ground truth来评估uplift model。

* uplift通常使用一些 `aggregated measures` ，例如 `uplift bins` 或者 `uplift curves`. 
* upliff-model的评估通常基于一种假设：相似 uplift-score 的人通常具有相同的行为（认为相同uplift-score的用户是同一类人。）



**Area Under Uplift Curves (AUUC)**

* 准备两个验证集：一个 treatment group验证集，一个 control group验证集。Userid 为 Primary-Key
* 拿训练好的模型预测两组用户 uplift-socres
* 两组用户根据uplift-scores进行降序排。该操作用于人群对齐。基于假设（相似uplift-scores的人具有相似的行为）
* 然后分别取两组的top 10%, 20%, ... 100%。计算两组的 转化率差异（并非直接用这个值，而是有一个诡异的公式。）。画出一个曲线
* 曲线下面积即为auuc
* 离线优化目标可以朝着auuc变大的方向去。





# Glossary

* quantile: 四分位，概率分布曲线下面积四等分，会得到的三个数
* decile：10分位，
* Percentile



# 参考资料

https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247498630&idx=1&sn=b36515e54c2dbc20186942102497c390&chksm=fbd749eacca0c0fc9e285ffc7d06e336115f387394362a4707c71377f02832f8c42bcc71cc7a&mpshare=1&scene=24&srcid=&sharer_sharetime=1585109170232&sharer_shareid=255a68ecb152bdfa3b164d51ce560a8d#rd

