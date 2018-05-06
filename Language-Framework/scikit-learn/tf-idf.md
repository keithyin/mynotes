# TF-IDF 与 scikit-learn



## TF （term frequency）

将一个 `document` 中的每一个 `term` 都赋予一个权重，最简单的方法就是将  `term` 在此 `document` 中出现的次数。用频数表示权重的方法叫做 TF，数学表示为：$\text{tf}_{t,d}$  ，表示 "document" `d` 中 "term" `t` 出现的次数。



## IDF（Inverse document Frequency）

使用 TF 方法进行 `query` 有个很严重的问题：所有的 `query term` 都被同等的对待。实际情况下，并不是所有的 `term` 都具有同样的辨别力。比如说，`and` 这个 `term` 几乎会出现在所有的 `document` 中。一个想法是，使用 `term` 的  `document` 间 `frequency` 来衰减 `term` 的 `document` 内 `frequency` 。`idf` 就是这么一个衰减系数。



$\text{df}_t$ : 集合中，包含 "term" `t` 的 `document` 个数

$\text{idf}_t = \log \frac{N}{\text{df}_t}$  : 其中 N 表示 集合中 `document` 的个数。  值越大，表示 "term" `t` 更具有辨别力。 



## tf-idf weighting

tf-idf weighting 的公式如下：


$$
\text{tf-idf}_{t,d} = \text{tf}_{t,d} *\text{idf}_t
$$


在 Query 情况下，如何计算document 的 Score：


$$
\text{Score}(q,d) = \sum_{t \in q} \text{tf-idf}_{t,d}
$$

## sklearn 与 tf-idf

先看 **CountVectorizer**

```python
from sklearn.feature_extraction.text import CountVectorizer
class CountVectorizer(input='content', 
                      encoding='utf-8', 
                      decode_error=’strict’, 
                      strip_accents=None, 
                      lowercase=True, 
                      preprocessor=None, 
                      tokenizer=None, 
                      stop_words=None, 
                      token_pattern='(?u)\b\w\w+\b',
                      ngram_range=(1, 1), 
                      analyzer='word', 
                      max_df=1.0, 
                      min_df=1, 
                      max_features=None, 
                      vocabulary=None, 
                      binary=False, 
                      dtype=<class ‘numpy.int64’>)
# 构造函数中设置: 计数属性
"""
raw_documents = ["what a beautiful day", "what is your name"]
"""
fit(raw_documents) # 开始统计 corpus 中的 token 的数量, 然后确定特征 token 都有啥
transform(raw_document) # 根据 fit 统计结果, 将 raw_documents 转成矩阵
get_feature_names() # 获取特征的名字
```



再看 **TfidfTransformer**

```python
# 给定 Count, 将其转化成 TFidf 
class TfidfTransformer(norm='l2', 
                       use_idf=True, 
                       smooth_idf=True, 
                       sublinear_tf=False)
"""
Counts:
[[3, 0, 1],
 [2, 0, 0],
 [3, 0, 0],
 [4, 0, 0],
 [3, 2, 0],
 [3, 0, 2]]
"""
fit(X) # 看一下整个 corpus
transform(X) # Counts 转成 tfidf 表示, 最终还会 Norm 一下.
```



最后看 **TfidfVectorizer**, **CountVectorizer与TfidfTransformer** 的结合体.

```python
vect = TfidfVectorizer(min_df=0, max_df=0.7,
                       analyzer='word',
                       ngram_range=(1, 2),
                       strip_accents='unicode',
                       smooth_idf=True,
                       sublinear_tf=True,
                       max_features=10
                      )
# max_features : 计算 corpus 的 term frequency，从大到小排，取前 max_features个。
```






## 参考资料

[https://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html](https://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html)

[https://nlp.stanford.edu/IR-book/html/htmledition/term-frequency-and-weighting-1.html](https://nlp.stanford.edu/IR-book/html/htmledition/term-frequency-and-weighting-1.html)

[https://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html](https://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html)

[http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

[http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)



