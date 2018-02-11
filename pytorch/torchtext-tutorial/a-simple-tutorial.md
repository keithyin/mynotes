# torchtext 教程

[翻译，原文链接](http://anie.me/On-Torchtext/)

## 概览

**torchtext.data**

* `torchtext.data.Example` : 用来表示一个样本，数据+标签
* `torchtext.vocab.Vocab`: 词汇表
* `torchtext.data.Datasets`: 数据集类，`__getitem__` 返回 `Example`实例
* `torchtext.data.Field` : 用来定义如何处理样本
  * 创建 `Example`时的 预处理
  * `batch` 时的一些处理操作。
* `torchtext.data.Iterator`: 迭代器，用来生成 `batch`




**torchtext.datasets**: 包含了常见的数据集.



## 基础

**Torchtext** 是一个非常强有力的库，她可以帮助我们解决 文本的预处理问题。为了能够更好的利用这个工具，我们需要知道她可以做什么，不可以做什么，也要将每个API和其我们想要的做的事情联系起来。另外一个值得夸赞的一点是，**Torchtext** 不仅可以和 **pytorch** 一起用，还可以和其它深度学习框架(tf,mxnet,...)。

下面是 text 预处理的工作列表，打勾的代表 **torchtext** 已经支持的工作：

- [x] **File Loading**: 加载不同文件格式的 `corpus`
- [x] **Tokenization**: 将句子 分解成 词
- [x] **Vocab**: 生成 词汇表
- [x] **Numericalize/Indexify**: 将 词 映射成 index
- [x] **Word Vector**: 词向量
- [x] **Batching**: generate batches of training sample (padding is normally happening here)
- [ ] **Train/Val/Test Split**: seperate your data into a fixed train/val/test set (not used for k-fold validation)（这个需要手动划分）
- [ ] **Embedding Lookup**: map each sentence (which contains word indices) to fixed dimension word vectors（这个可以使用 pytorch 的 Embedding Layer解决）




下面是对以上任务一个直观的表述：

```python
"The quick fox jumped over a lazy dog."
-> (tokenization) 
["The", "quick", "fox", "jumped", "over", "a", "lazy", "dog", "."]

-> (vocab)
{"The" -> 0, 
"quick"-> 1, 
"fox" -> 2,
...}

-> (numericalize/indexify)
[0, 1, 2, ...]

-> (embedding lookup)
[
  [0.3, 0.2, 0.5],
  [0.6, 0., 0.1],
  ...
]
```

这些过程非常容易搞砸，特别是 `tokenization`。研究者们经常花费大量的时间编写代码来处理这些问题。**Torchtext** 将这些常用的预处理操作整理起来，使得更加好用。







## 一个简单例子

NLP 的一般数据预处理流程为：

1. 定义样本的处理操作。---> `torchtext.data.Field`
2. 加载 corpus  （都是 string）---> `torchtext.data.Datasets` 
   * 在`Datasets` 中，`torchtext` 将 `corpus ` 处理成一个个的 `torchtext.data.Example` 实例
   * 创建 `torchtext.data.Example` 的时候，会调用 `field.preprocess` 方法
3. 创建词汇表， 用来将 `string token` 转成 `index`  ---> `field.build_vocab()`
   * 词汇表负责：`string token ---> index`,  `index ---> string token` ，`string token  ---> word vector`
4. 将处理后的数据 进行 batch 操作。---> `torchtext.data.Iterator`
   * 将 `Datasets` 中的数据 `batch` 化
   * 其中会包含一些 `pad` 操作，保证一个 `batch` 中的 `example` 长度一致
   * 在这里将 `string token` `index`化。



`tokenization，vocab， numericalize， embedding lookup` 和数据预处理阶段的对应关系是：

* `tokenization` ---> `Dataset ` 的构造函数中，由 `Field` 的 `tokenize` 操作
* `vocab` ---> `field.build_vocab` 时，由 `Field` 保存 映射关系
* `numericalize` ---> 发生在 `iterator` 准备 `batch` 的时候，由 `Field` 执行 `numericalize` 操作
* `embedding lookup` ---> 由 `pytorch Embedding Layer` 提供此功能。 





**首先，我们要创建 Field 对象：** 这个对象包含了我们打算如何预处理文本数据的信息。 她就像一个说明书。下面定义了两个 **Field** 对象。

```python
import spacy
spacy_en = spacy.load('en')

def tokenizer(text): # create a tokenizer function
    # 返回 a list of <class 'spacy.tokens.token.Token'>
    return [tok.text for tok in spacy_en.tokenizer(text)]

TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
LABEL = data.Field(sequential=False, use_vocab=False)
```

如果想要整型 LABEL， 就需要将 `use_vocab=False`. **Torchtext** 可能也会允许使用 text 作为 label，但是现在我还没有用到。



然后我们可以通过 `torchtext.data.Dataset` 的类方法 `splits` 加载所有的语料库：(假设我们有三个语料库，`train.tsv, val.tsv, test.tsv`)

```python
train, val, test = data.TabularDataset.splits(
        path='./data/', train='train.tsv',
        validation='val.tsv', test='test.tsv', format='tsv',
        fields=[('Text', TEXT), ('Label', LABEL)])
```



然后加载预训练的 `word-embedding`

```python
TEXT.build_vocab(train, vectors="glove.6B.100d")
```

我们可以直接传一个 string，然后后端会下载 word vectors 并且加载她。我们也可以通过 `vocab.Vectors` 使用自定义的 vectors. 



下一步将要进行 **batching** 操作：用 torchtext 提供的 API 来创建一个 iterator

```python
train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test), sort_key=lambda x: len(x.Text),
        batch_sizes=(32, 256, 256), device=-1)

batch = next(iter(train_iter))
print("batch text: ", batch.Text) # 对应 Fileld 的 name
print("batch label: ", batch.Label)
```

需要注意的是，如果您要运行在 CPU 上，需要设置 `device=-1`， 如果运行在GPU 上，需要设置`device=0` 。您可以很容易的检查 batch 后的结果，同时会发现，`torchtext` 使用了动态 `padding`，意味着 `batch`内的所有句子会 `pad` 成 `batch` 内最长的句子长度。

`batch.Text` 和 `batch.Label` 都是 `torch.LongTensor` 类型的值，保存的是 `index` 。

如果我们想获得 word vector，应该怎么办呢？ 

* `Field` 的 `vocab` 属性保存了 word vector 数据，我们可以把这些数据拿出来
* 然后我们使用 Pytorch 的 `Embedding Layer` 来解决 `embedding lookup` 问题。 

```python
vocab = TEXT.vocab
self.embed = nn.Embedding(len(vocab), emb_dim)
self.embed.weight.data.copy_(vocab.vectors)
```





