# Structured Prediction Basics

**结构预测问题(`structured prediction task`)：输出是结构数据**



## 预测问题

* 二分类：给定输入 ，输出 `0/1` 分类
* 多分类：给定输入，输出 多分类结果
* 结构化预测： 给定输入，输出的标签可能有无限种（多 label 分类）
  * part-of-speech tagging
  * 翻译



## 如何训练 Structured Models

**Teacher Forcing**

* just feed in the correct previous tag.



**Teacher Forcing存在的几个问题**

* `teacher forcing` 假设 feed 正确的 previous input， 但是在测试阶段，我们不知道正确的 previous input。这样就会导致 训练和测试的情况不一致。
* **Exposure Bias**: 在训练的时候，没有暴露错误 给 模型， 所以在测试的时候，模型并不知道该如何处理这个问题。
* **Label Bias** :   ?????????????????? 不明白



**Locally Normalized Models**

* `each decision` made by the model has a probability that adds to one

**Globally Normalized Models (energy-based models)**

* `each sentence` has a score, which is not normalized over a particular decision

