# 决策树



## 特征选择

* 防止模型过深
* 防止模型过宽



**信息增益（information gain）**

> 原数据集的熵 减去 划分后的熵， 增益越大越好，即：划分后的熵越小越好


$$
\text{Gain}(D, a) = \text{Ent}(D) - \sum_{v=1}^V\frac{|D^v|}{|D|} \text{Ent}(D^v)
$$
**信息增益比**

> 信息增益模型对取值比较多的模型会有偏好，这种偏好会带来不利影响，
>
> 1. 模型过于复杂
> 2. 过拟合
>
> 举例：假设有个任务，根据今天的天气情况来推测明天的天气。其中有个日期特征，那么这个日期特征一定是个信息增益最大的目标，但是日期却不能作为到底是啥天气的评价标准。
>
> 信息增益比对 取值数目较少的属性有所偏好。


$$
\text{Gain_ratio(D,a)} = \frac{\text{Gain(D, a)}}{\text{IV(D, a)}}
$$
$\text{IV(D,a)}$ 即 特征 $a$ 作为随机变量的熵。



**基尼指数**

> 数据集 D 的纯度可用基尼值来度量

$$
\text{Gini(p)} = \sum_{k=1}^K p_k(1-p_k) = 1-\sum_{k=1}^Kp_k^2
$$

* 有 $K$ 个类，第 $k$ 个类的概率是 $p_k$ 。

$$
\text{Gini_index(D,a)} = \sum_{v=1}^V \frac{|D^v|}{|D|} \text{Gini}(D^v)
$$

* `Gini_index` 越低越好。



## 剪枝策略

* 预剪枝
  * 决策树生成过程中，对每个节点的划分进行估计，如果无法提供更好的泛化性能，放弃。
* 后剪枝
  * 先生成一个完整的决策树，然后再自底向上的进行剪枝。
  * 看剪枝后，能否提升验证集精度



## 特征处理

**连续值处理**

* 量化 + 二分法
  * 将特征的所有值从小到大排序，去中间的值

**缺失值处理**





## 随机森林

**random everything**

* 样本扰动
* 属性扰动



## CART

可用于分类，也可用于回归问题

**分类树使用基尼指数进行特征选择**



**回归树**

```python
# 如何进行特征选择和切分

while mse <= threshold:
  for feat in feats:
      for split_v in range(feat_min, feat_max, step=step):
          # 特征值小于 split_v 的放在一起， 特征值大于 split_v 的放在一起
          # 分别对 y 求平均，然后求平方误差
          pass
      pass
  # 找到 平方误差最小的 feat 和 split


```





## BDT

**Boosting Decision Tree**

* 决策树的加法模型

$$
f_M(x) = \sum_{m=1}^M T(x;\theta_m)
$$

* 树中的每个节点都代表了这个节点上的数据集的 平均值。
* 树结点不停的在学习残差。



**算法流程**

* 输入数据
* 启发式方法计算切分属性和切分点
* 切分数据，计算均值，计算残差，用残差作为数据的 label。
* 返回第一步

**特点：**

* 除了 根节点，其它节点都有个 值。
* 残差使得他们聚在一起， 节点代表的值由 最小化节点的 loss 获得，mse loss 就是均值咯。



## GBDT

[https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d)


$$
\begin{aligned}
\text{Loss}(y, f_{m-1}(x)) \\ 
\frac{\partial \text{Loss}(y, f_{m-1}(x))}{\partial f_{m-1}(x)}
\end{aligned}
$$
为梯度拟合一个回归树，

* 梯度使得他们聚在一起，节点代表的值  由最小化节点的 loss 获得(使用线性搜索)。
* 梯度相近意味着什么，