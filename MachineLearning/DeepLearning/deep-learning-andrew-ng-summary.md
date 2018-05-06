# 超参数调试，正则化以及优化 总结



## 超参数

* layers
* hidden units
* learning rates
* activation functions
* ...



**如何快速的寻找合适的超参数：**

* train set （用来训练模型）
* dev set (val set) （用来评估训练模型的 好坏）
* test set



**mismatched train/test distribution**

* 确保， dev set 和 test set 来自同一数据分布
* ​



## Bias Variance

>  不是机器学习中的那个 bias variance



* 高方差： over-fitting
* 高偏差： under-fitting



* 高方差： 训练集误差小，验证集误差大
* 高偏差： 训练集误差大（与人类表现相比），训练集误差与验证集误差相差不大
* 高偏差，高方差：
* 低偏差，低方差：

