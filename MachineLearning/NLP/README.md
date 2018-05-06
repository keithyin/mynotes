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










## Terminology

* syntactic : 语法
* semantic：语义
* part-of-speech tagging : 词性标注
  * 动词/名词/副词.... 
* ​Noun Phrase ： 名词性短语

