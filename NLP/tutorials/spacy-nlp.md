# 使用 spacy 进行自然语言处理



## 介绍

自然语言处理(NLP) 是人工智能方向一个非常重要的研究领域。 自然语言处理在很多智能应用中扮演着非常重要的角色，例如：

*  `automated chat bots`,
*  `article summarizers`,
*  `multi-lingual translation`
*  `opinion identification from data`

每一个利用NLP来理解非结构化文本数据的行业，不仅要求准确，而且在获取结果方面也很敏捷。

自然语言处理是一个非常广阔的领域，NLP 的任务包括

*  `text classification`,
*  `entity detection`,
*  `machine translation`, 
* `question answering`, 
* `concept identification`. 

在本文中，将介绍一个高级的 NLP 库 - spaCy 



## 内容列表

1. 关于 spaCy 和 安装
2. Spacy 流水线 和 属性
   1. Tokenization 
   2. Pos Tagging
   3. Entity Detection
   4. Dependency Parsing
3. 名词短语
4. 词向量
5. 集成 Spacy 和 Machine Learning
6. 与 NLTK 和 coreNLP 的对比



## 1.关于 spaCy 和 安装

### 1.1 关于 Spacy

Spacy 是由 `cython` 编写。因此它是一个非常快的库。 `spaCy` 提供简洁的接口用来访问其方法和属性 governed by trained machine (and deep) learning models.

### 1.2 安装

安装 `Spacy`

```shell
pip install spacy
```

下载所有的数据和模型

```
python -m spacy.en.download all
```

现在，您可以使用 `Spacy` 了。



## 2. Spacy 流水线 和 属性

要想使用 `Spacy` 和 访问其不同的 `properties`， 需要先创建 `pipelines`。 **通过加载 模型 来创建一个 pipeline**。 `Spacy` 提供了许多不同的 [模型](https://github.com/explosion/spacy-models/) , 模型中包含了 语言的信息- 词汇表，预训练的词向量，语法 和 实体。

下面将加载默认的模型- `english-core-web`

```
import spacy 
nlp = spacy.load(“en”)
```

`nlp` 对象将要被用来 创建文档，访问语言注释和不同的 nlp 属性。**我们通过加载一个 文本文件 来创建一个 document** 。I am using reviews of a hotel obtained from tripadvisor’s website. The data file can be downloaded [here](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/04/04080929/Tripadvisor_hotelreviews_Shivambansal.txt).

```
document = unicode(open(filename).read().decode('utf8')) 
document = nlp(document)
```

The document is now part of spacy.english model’s class and is associated with a number of properties. The properties of a document (or tokens) can listed by using following command:

```
dir(document)
>> [ 'doc', 'ents', … 'mem']
```

This outputs a wide range of document properties such as – tokens, token’s reference index, part of speech tags, entities, vectors, sentiment, vocabulary etc. Let’s explore some of these properties.









## 参考资料

[https://github.com/pytorch/text](https://github.com/pytorch/text)

[https://www.analyticsvidhya.com/blog/2017/04/natural-language-processing-made-easy-using-spacy-%E2%80%8Bin-python/](https://www.analyticsvidhya.com/blog/2017/04/natural-language-processing-made-easy-using-spacy-%E2%80%8Bin-python/)

