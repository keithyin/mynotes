# Language Modeling

[http://www.cs.columbia.edu/~mcollins/lm-spring2013.pdf](http://www.cs.columbia.edu/~mcollins/lm-spring2013.pdf)



* problem
  * construction a **language model** from **a set of example sentences** in a language.
* task
  * 假设我们有 一个 corpus（句子的集合）
  * 用 这个 corpus 进行 parameter estimating
* language model 的定义
  * 一个有限集合 $V$ 表示某语言中所有 `word`
  * 一个无限集合表示 $S$ 某**语言(并不只是corpus中的句子)**中的所有句子  , 每个句子长度可能不同
  * 对于任何 $<x_1, ..., x_n> \in S, p(x_1, ..., x_n)\ge 0$
  * 且 $\sum_{s\in S}p(s) = 1, s=<x_1, ..., x_n>$



## 为什么需要 language model

* 给一些任务提供 先验概率 $p(x_1, ..., x_n)$
  * 比如用在 语音识别上，来挑更符合 语言模型的 candidates
  * 比如用在翻译任务上，来挑更符合 语言模型的 candidates



## 最简单的一个模型

$$
p(x_1, ..., x_n) = \frac{c(x_1, ..., x_n)}{N}
$$

* $c(x_1, ..., x_n)$ : 句子 $<x_1, ..., x_n>$ 在 corpus 出现的次数
* $N$ : corpus 中的总句子数



优点：

* 简单

缺点：

* 泛化太差，corpus 中没有出现的句子就会被设置为 0



## Markov Models



### Markov Models for Fixed-length Sequences

* first order Markov process， first-order Markov assumption
  * $p(x_1, x_2, ..., x_N) = \prod_{n=1}^N p(x_n|x_{n-1})$
  * 引入 $x_0$ 这样更好写表达式
* second-order Markov process
  * $p(x_1, x_2, ..., x_N) = \prod_{n=1}^N p(x_n|x_{n-1}, x_{n-2})$
  * 假设 $x_0=x_{-1}=<start>$

