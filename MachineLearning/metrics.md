# 机器学习中一些评估标准

## 二分类

**Precision（精确率）**

> 挑出来 **认为是 Positive** 的中，**Positive** 的概率

$$
\text{P} = \frac{TP}{TP+FP}
$$



**Recall (召回率)**

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
* 面积越大表示：分类效果越好，因为面积能够表示 `Positive 最低分` 与 `Negative 最高分` 的距离。

