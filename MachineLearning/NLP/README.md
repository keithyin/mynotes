# NLP (自然语言处理)



## 任务

* sentences matching (两个句子是否匹配)
* **Paraphrase(释义) identification** aims to determine whether two sentences have the same meaning.
* textual entailment：
  * entailment： 蕴含，包含的意思
  * 这歌文字不长，却蕴含着丰富的内容
* retrieval
* semantic parsing
* sentence modeling



**classification tasks**

* subjectivity
* question-type classification: 问题类型分类。


* sentiment analysis (情感分析)
* paraphrase identification/detection : 
  * It is usually formalized as a binary classification task: for two sentences (S1 , S2 ), determine whether they roughly have the same meaning.
  * 即：看看两个句子是不是同一个意思
  * 可以看作 二分类 问题， yes/no
* semantic similarity :语义相似性
  * 两个句子的相似程度
  * 多分类或者回归问题
* entailment recognition
* summarization：文本摘要
* discourse analysis : [http://www.english.ugent.be/da](http://www.english.ugent.be/da)
* machine translation : 
* grounded language learning : 
* image retrieval



## 如何建模 句子对

* siamese network
* convolutional matching model : 




## 词性标注 (Part-of-speech tagging)

```python
noun ： 名词
verb ： 动词
adjective ： 形容词
adverb ： 副词
pronoun ： 代词
preposition ： 介词
conjunction ： 连词
interjection ： 感叹词
article or (more recently) determiner ： 冠词
```



## Evaluation

**perplexity**

* e 的 交叉熵损失次方



**BLEU**

* 比较 n-gram 的重叠。 用来衡量 系统输出的结果和 真实目标的匹配度。

```python
# reference : Taro visited Hanako
# system : the Taro visited the Hanako

# 1-gram : 3/5
# 2-gram: 1/4
# brevity = min(1.0, |system|/|reference|)
# BLEU-2 = (3/5*1/4)^(1/2) * brevity
```

* https://towardsdatascience.com/bleu-bilingual-evaluation-understudy-2b4eab9bcfd1
* https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213
* 计算过程
  * 通常结果取的是多个 `-gram` 的平均值, 比如`1-gram`, `2-gram`, `3-gram`, `4-gram`的平均值
  * 一个是 modified precision
  * 第二个是 brevity penalty
  * 然后一个大公式.
    * $BP$: brevity penalty, 如果翻译的长度大于 reference的长度取值1, 否则取值$e^{(1-r/c)}$
    * $N$ : 表示有 N个 reference, $w_n$ 表示每个 reference 的权重
    * $p_n$ : 就是 modified precision (n-gram 的 score 了)

$$
BLEU=BP*\exp\Bigr(\sum_{n=1}^Nw_n*\log p_n \Bigr)
$$

* 如何计算 modified precision
  * 将candidate 和 reference 进行 n-gram 切分
  * 然后统计每个n-gram出现的频数
  * 要计算一个除法, 其中分母是 candidate 中 n-gram 的个数(不是去重后的哦)
  * 分子为 $\sum_{v\in cantidate} \min(candidate_v, \max(r1_v, r2_v,...))$
    * $v$ : 为 candidate 中的 n-gram(去重一下)
    * $candidate_v$ 为 $v$ 在 $candidate$ 中出现的频次
    * $rn_v$ : 为 $v$ 在第n 个 reference 中出现的频次.
  * 这么一除就计算得到了modified-precision






## Terminology

* syntactic : 语法
* semantic：语义
* part-of-speech tagging : 词性标注
  * 动词/名词/副词.... 
* Noun Phrase ： 名词性短语

