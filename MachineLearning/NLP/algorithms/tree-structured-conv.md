# Dependency-based Convolutional Neural Networks for Sentence Embedding

**word-embedding + syntactic parse trees**



**大概意思是：**

* 卷积还是卷积，重新排列一下 单词的组织方式。
* ​





## 摘录

> To combine deep learning with linguistic structures, we propose a dependency-based convolution approach, making use of tree-based n-grams rather than surface ones, thus utlizing nonlocal interactions between words.

* dependence-based conv， 有效的利用句子的 语法信息





> CNNs, being invented on pixel matrices in image processing, only consider **sequential n-grams** that are consecutive on the surface string and **neglect long-distance dependencies**, while the latter play an important role in many linguistic phenomena such as 
>
> * negation, subordination, and wh-extraction,
>
> all of which might dully affect the sentiment, subjectivity, or other categorization of the sentence.



> tree n-grams are significantly sparser than surface n-grams

* tree n-grams 是什么样的？？？？