# 机器学习中一些评估标准

## 二分类

**Precision（精确率）模型认为是正例的样本中，有多少是真实正确的**

> 挑出来 **认为是 Positive** 的中，**Positive** 的概率

$$
\text{P} = \frac{TP}{TP+FP}
$$



**Recall (召回率)， 模型认为是正例的样本中，有多少是真实正例 : 数据库中的正例**

> 挑出来的 **Positive** 占所有 **Positive** 的比重

$$
\text{R} = \frac{TP}{TP+FN}
$$

**$F_\beta$**
$$
\text{F}_\beta = \frac{(\beta^2+1)*P*R}{\beta^2*P + R}
$$
**[Roc-Auc Score](http://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics)**

> performance of a binary classifier system as its discrimination threshold is varied. 

* 当 阈值变化时， `TP-rate` 与 `FP-rate` 构成的曲线下面的面积
* 面积越大表示：分类效果越好，因为面积能够表示 `Positive 最低分` 与 `Negative 最高分` 的距离。为什么这么说呢？ 考虑极端情况，正例的 score 为 1, 负例的 score 为 0, 那么 面积为 1, 其它情况都会降低。


`TP-rate = 模型认为的正例是正例个数 / 真实的正例个数`

`FP-rate = 模型认为的正例不是正例的个数 / 真实的正例个数 `





## 检索

**mAP**

