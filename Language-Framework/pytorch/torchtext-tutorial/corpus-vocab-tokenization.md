# torchtext

本文主要目的是搞清楚 **torchtext** 是如何实现以下几个步骤的：

*  通过 `corpus` 构建  `Vocab`
*  从 预训练的词向量中，将 `Vocab` 中包含的 `token` 的向量加载进来。
*  给定文本，如何通过 `Vocab` 对其进行 `numericalize` 。




使用 **torchtext** 进行文本预处理的流程如下所示：

```python
# set up fields
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)

# make splits for data
# 这里主要是用到了 Field 的 tokenization 功能来创建 Example
train, test = datasets.TREC.splits(TEXT, LABEL, fine_grained=True)

# print information about the data
print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))

# build the vocabulary
# train 中保存了 corpus 的所有 token，通过 token 的统计信息来创建 Vocab
# 还可以传入更多的参数来控制 Vocab 的构建
TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
LABEL.build_vocab(train)

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))
print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

# make iterator for splits
# 创建 iterator，在迭代器中使用 Field 提供的 numericalize 和 pad 方法来对 batch example 进行
# 数值化和 填充。
train_iter, test_iter = data.BucketIterator.splits(
    (train, test), batch_size=3, device=0)
```



**通过 corpus 构建 Vocab**

* 统计 `corpus` 中的 所有 `token`
* 通过 `min_freq, max_size`, 来选择忽略一些低频的 `token`
* 然后再添加一些 特殊的 `token ` ，`<unk>, <pad>` 
* 用这些 `token` 来构建 `Vocab`



**从预训练的 词向量中将 Vocab 中 token 对应的词向量加载过来**

* 如果 `Vocab` 中的 `token` 可以在 预训练的词向量中找到，则加载
* 如果找不到 ，用 `0` 初始化。



**给定文本，通过 Vocab 进行 numericalize**

* 给定的 文本 `token`， 如果可以在 `Vocab` 中找到，则赋予其对应的 `index`
* 否则， 返回值 `0` （`<unk>` 的 `index` ）



