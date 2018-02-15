# SpaCy 类结构介绍

`SpaCy` 中对文本信息的抽象成三个类：

*  `Doc`:
  * A `Doc` is a sequence of `Token` objects. Access sentences and named entities, export annotations to numpy arrays, losslessly serialize to compressed binary strings. The `Doc` object holds an array of `TokenC` structs. The Python-level `Token` and `Span`  objects are views of this array, i.e. they don't own the data themselves.
* `Span`
  * A slice from a `Doc` object.
* `Token`
  * An individual token — i.e. a word, punctuation symbol, whitespace, etc.





## 参考资料

[https://spacy.io/api/doc](https://spacy.io/api/doc)

