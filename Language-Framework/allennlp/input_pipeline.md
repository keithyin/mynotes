# input pipeline

* tokenization
* vocab
* neumericalization
* batching (padding)




## 核心类

* `Tokenizer`:  `allennlp.data.tokenizers` 
  * 执行 `tokenization`
* `Token` : `allennlp.data.tokenizers.Token` 
  * 用来表示一个 符号（单词/词语）
* `TokenIndexer` : `allennlp.data.token_indexers`，
  * 执行 `neumericalization`
* `Field`: `allennlp.data.fields`
  * 用来表示样本的不同域（输入域，输出域）
* `Instance` : `allennlp.data`
  * 用来表示一个样本实例，样本由多个 `Field` 构成
* `Batch` : `allennlp.data.dataset`
  * 构建 mini-batch
  * 计算如何填充 batch 中的样本，然后调用
* `Vocabulary` : `allennlp.data.vocabulary`
  * 保存 `token->idx` , `idx->token` 之间的映射关系
* `Iterator` : `allennlp.data.iterators`
  * 迭代器，用来迭代一个又一个mini-batch



**在使用 allennlp 输入流水线的时候会分成三步走，下面也将从三个部分详细介绍 allennlp 中的 输入流水线**

* 自定义 `DatasetReader` 类
* 构建 `Vocabulary` 实例
* 构建迭代器



## 自定义 `DatasetReader` 类

* 负责从文本文件中读数据，并将其中的每个样本转成 `Instance` 实例
* 涉及到类型有:
  * `TokenIndexer`
  * `Token`
  * `Field`
  * `Instance`
* 必须重写其 `_read` 方法。负责 `yield Instance` 实例
```python
class PosDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like
        The###DET dog###NN ate###V the###DET apple###NN
    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                pairs = line.strip().split()
                sentence, tags = zip(*(pair.split("###") for pair in pairs))
                yield self.text_to_instance([Token(word) for word in sentence], tags)
```

* 注意：`TokenIndexer` 是数据集级别的。`Field` 是样本级别的
* 一个 `Instance` 实例由多个 `Fields` 实例构成，一个 `Fields` 又可能存在多个 `TokenIndexer` 。



## 构建 Vocabulary

```python
reader = PosDatasetReader()
train_dataset = reader.read(cached_path(
    'https://raw.githubusercontent.com/allenai/allennlp'
    '/master/tutorials/tagger/training.txt'))
validation_dataset = reader.read(cached_path(
    'https://raw.githubusercontent.com/allenai/allennlp'
    '/master/tutorials/tagger/validation.txt'))
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
```

* `Vocabulary.from_instances`
  * `instance.count_vocab_items(self, counter: Dict[str, Dict[str, int]])`
    * 对 `instance` 中的每个 `field` 调用 `field.count_vocab_items(self, counter: Dict[str, Dict[str, int]])`
      * 对`field`中的每一个 `indexer`调用 `indexer.count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]])`
* 其中由 `Dict[str, Dict[str, int]]` 表示的是

```json
{
  "indexer1.namespace":{"token1":count1, "token2":count2, ...},
  "indexer2.namespace":{"token1":count1, "token2":count2, ...}
}
// 如果 不同的 indexer 的 namespace 相同，那么 counter 会当作同一个 indexer看待
```

* 在`Vocabulary` 实例中保存的映射关系也是 `namespace` 相关的



## 构建 `Iterator`

```python
iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])
iterator.index_with(vocab)
```



产生最终 训练数据（`tensor`） 的具体流程为：

* 通过 `indexer.tokens_to_indices(self.tokens, vocab, indexer_name)` ，将 `tokens` 转成 `indices`
* 通过 `indexer.pad_token_sequence(self, desired_num_tokens, padding_lengths)` 对 `indices` 进行填充



## Tokenizer

* 将 `string` 搞成 `list of tokens`


## Token

* 用来表示**一个**  `token`



## TokenIndexer

* 一个 `TokenIndexer` 是一个 `namespace` , `namespace`在token计数的时候非常有用。
* 负责给 `token` 设置 `index` 的工具类
* 二级 `token` 也可以由其完成

**功能**

* 统计 `token` 在语料库中出现的次数:

```python
def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]])
```

* 将 `token list` 转成 `token index list`，可以添加开始 `token:<GO>`  和结束 `token:<END>`


```python
def tokens_to_indices(self,
                      tokens: List[Token],
                      vocabulary: Vocabulary,
                      index_name: str) -> Dict[str, List[List[int]]]
"""
根据 vocab 将 token 转成 index list
"""
```

* 将 单个 `token list` 填充到期望的长度。**在 char 级别的数据输入时有用**


```python
def pad_token_sequence(self,
                       tokens: Dict[str, List[int]],
                       desired_num_tokens: Dict[str, int],
                       padding_lengths: Dict[str, int]) -> Dict[str, List[int]]:
	"""
	tokens: {"tokens": [2,1,3,4]}
	"""
    return {key: pad_sequence_to_length(val, desired_num_tokens[key])
                for key, val in tokens.items()}
```



## Field

* 用来区分样本中的不同部分（输入域，输出域）
* 构成：`tokens, token_indexers`
  * `tokens` : list of tokens，用来表示一个句子
  * `token_indexers` ：用来负责将 `token` 转成 `index`

**功能**

* 由于单个 token 也可能由多个 字母构成，此函数可用来填充
* `get_padding_lengths(self)`
  * 返回 `field` 中各个元素的长度




## Instance

* 多个 `Field` 的集合，一个 `Field` 表示样本的一个字段


**方法**

* `count_vocab_items(self, counter: Dict[str, Dict[str, int]])`: 
  *  会调用 `field` 的 `count_vocab_items(self, counter: Dict[str, Dict[str, int]])` 方法更新 `counter`
* `index_fields(self, vocab: Vocabulary)`: 
  * 会调用 `field.index(vocab)` 
  * 通过 `vocab` 将 `token` 转成 `index`
* `get_padding_lengths(self)` 
  *  调用了 `field.get_padding_lengths()`
* `as_tensor_dict(self)` ，返回的字典如下所示，会执行 `padding` 操作

```json
{"field_name_1": {
  "indexer_name_1": tensor, 
  "indexer_name_2": tensor},
 "field_name_2":{
   "indexer_name_3": tensor,
   "indexer_name_4": tensor
 }
}
```



## Batch

* 用来构建 mini-batch
* 负责填充




## Vocabulary

* 保存了 `string -> idx` 和 `idx -> string` 的映射
* `Vocabulary` 中存在 `namespace` 的区分，不同的 `namespace` 相当于属于不同的 `Vocab`，比如：中英翻译数据集，一个`namespace`就可以是 `chinese`，另一个 `namespace` 是 `english`。


**方法**

* 在构建 `vocab` 的时候会调用 `instance.count_vocab_items()` 方法




## allennlp 输入流水线的整个流程

