# roc-auc score

## ROC 曲线

要想搞清楚 ROC 曲线是什么意思，首先要明白两个概念：

* `TPR` (true positive rate): $TPR=\frac{TP}{TP+FN}$
* `FPR` (false positive rate) : $FPR=\frac{FP}{FP+TN}$



**ROC** 曲线就是以 `TPR` 为 `y-axis`， `FPR` 为 `x-axis`， 当二元分类器的阈值变化时，在坐标轴上打的点连成的线。





## 参考资料

[http://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics](http://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics)

