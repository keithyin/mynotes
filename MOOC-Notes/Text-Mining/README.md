[课程地址](https://www.coursera.org/learn/text-mining/home/welcome)

# week2



### 2.5 Topic Mining and Analysis

* topic: main idea discussed in text data
* Topic Mining 如何做
  * 挖掘 topics
  * 然后 确认 document 包含哪些 topics
* 问题的输入与输出
  * 输入
    * N 个文本文件 $C={d_1, d_2, ..., d_N}$
    * 多少个 topic: $k$ ,指出数量就好了
  * 输出:
    * k个 topics :$\{\theta_1, ..., \theta_k\}$
    * 输入文档的 topic 分布 $\{\pi_{n1}, ... \pi_{nk}\}$

方法:

* using the term(term, phrase) as topic
  * 如何挖掘 topics
    * 从 corpus 中找词, 然后给词评分 (频率, TF-IDF, 来给topics 评分)
    * 根据实际情况选择 topic
  * 如果计算 文档 coverage (topic的分布)
    * 继续计算频数
  * 缺陷
    * ...
* using the term distribution to representation topic
  * 生成模型做文本挖掘
    * 首先为数据设计一个模型, 表示数据是如何生成的
    * 