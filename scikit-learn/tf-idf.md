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




## 参考资料

[https://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html](https://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html)

[https://nlp.stanford.edu/IR-book/html/htmledition/term-frequency-and-weighting-1.html](https://nlp.stanford.edu/IR-book/html/htmledition/term-frequency-and-weighting-1.html)

[https://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html](https://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html)



