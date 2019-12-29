# input pipeline

* `tokenization` : 
* `vocab` : 
* `neumericalization` : `TokenIndexer` 负责统计`token frequency` 和 `neumnaricalization`
* `batching (padding)` :



## 核心类介绍

* `Tokenizer`:  `allennlp.data.tokenizers` 
  * 执行 `tokenization`
* `Token` : `allennlp.data.tokenizers.Token` 
  * 用来表示一个 符号（单词/词语）
* `TokenIndexer` : `allennlp.data.token_indexers`，
  * 通过传入的 `Vocab` 来执行 `neumericalization` 
  * 该类其实只是提供了一些 `helper function`, 其并 **没有保存什么状态**
  * `Indexer` 的 `namespace` 是用来区分 `Counter` 和 `Vocab` 的, 不同的 `namespace` 的统计完全独立.
* `Field`: `allennlp.data.fields`
  * 用来表示样本的不同域（输入域，输出域）
  * `Field` 里面是有个 `get_padding_lengths` 的, 这个是用来 细粒度的 填充的(比如 `char-level`)
  * 由于对于同一个 `Field` 可能会有不同的 `Indexer` 方式, 比如: 既有`token-level` , 也有 `char-level` 的`indexer` , 所以在创建 `Field` 对象的时候, 有个参数为`Dict[Indexer]`, 这个用来用来对该`Field` 执行不同 `Indexer` 操作的.
  * `Field` 的构建包含 `Field` 的 `Tokens` 和 `Dict[str, Indexer]` 
  * 和 `torchtext` 不同, `torchtext` 的 `Field` 表示的是整个数据集的 `Field`, 而 `allennlp` 的 `Field` 表示的仅是 一个 `instance` 的 `field`
* `Instance` : `allennlp.data`
  * 用来表示一个样本实例. 一个样本实例通常是由多个 `Field` (`LableField, TextField`)构成, 所以`Instance` 对象由 `Dict[str, Field]` 构建, 其中`dict` 的 `key` 用来表示 `Field` 的名字.
* `Batch` : `allennlp.data.dataset`
  * 构建 `mini-batch`
  * 计算如何填充`batch` 中的样本，然后调用
  * 这里面也有个 `get_padding_lengths`, 这个是 `batch` 级别的填充
* `Vocabulary` : `allennlp.data.vocabulary`
  * 保存 `token->idx` , `idx->token` 之间的映射关系
* `Iterator` : `allennlp.data.iterators`
  * 迭代器，用来迭代一个又一个 `mini-batch`



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
      	# 一个 field 可以有多个indexer, 比如 既有 char-level, 也有 word-level
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

* 只是构建 `Vocabulary` , `tokenization` 并不是在这操作的.

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
// 如果 不同的 indexer 的 namespace 相同，那么 counter 会当作同一个indexer看待
// 不同的 indexer 共享相同的 token counter? 这个是什么情况?
```

* 在`Vocabulary` 实例中保存的映射关系也是 `namespace` 相关的



## 构建 `Iterator`

```python
# 这个地方是和 Dataset 分离的
iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])
iterator.index_with(vocab)
```

产生最终 训练数据（`tensor`） 的具体流程为：

* 通过 `indexer.tokens_to_indices(self.tokens, vocab, indexer_name)` ，将 `tokens` 转成 `indices`
* 通过 `indexer.pad_token_sequence(self, desired_num_tokens, padding_lengths)` 对 `indices` 进行填充

## Tokenizer

* **这个似乎在 `Token-Level` 的模型上并不常用.**
* 将 `string` 搞成 `list of tokens`
* 负责: splitter, 构建 `List[Token]`, 负责 `truncation`
* `Splitter`: 进行 `split` 的核心, 会被 `Tokenizer 包含进去`
  * `Splitter` : 负责输出 `List[Token]`


## Token

* 用来表示**一个**  `token`, 主要用到的两个属性可能就是`Token.text, Token.idx` 了.

## TokenIndexer

* 三大功能: `token 计数, token->index, 填充`
* 一个 `TokenIndexer` 是一个 `namespace` , `namespace`在`token` 一个 `namespace` 表示一类 `字典`。
  * 多个`TokenIndexer`也可以具有相同的 `namespace` 的
* 负责给 `token` 设置 `index` 的工具类
  * 二级 `token` 也可以由其完成, 比如 `char-level` 的数据, 可以使用 `TokenCharactersIndexer`
  * `char-level` 的时候, 在构建 `field` 的时候与 `word-level` 一致, 只需要将 `indexer` 换成 `TokenCharactersIndexer` 就好了
* 重点
  * **只是一个工具类**, 其中并没有包含什么重要的属性. 在进行 `tokenization` 的时候 还是需要 `Vocabulary` 

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

* 在 `Indexer` 中, 一个比较重要的是 `namespace` 的概念. 往深一步看代码的时候, 可以发现计数器`counter` 是个`Dict[namespace, Dict[str, int]]` , 这就比较容易理解了. 

## Field

* 用来区分样本中的不同部分（输入域，输出域）
* 构成：`tokens, token_indexers` !!!!!!!!!!
  * `tokens` : list of tokens，用来表示一个句子
  * `token_indexers` ：用来负责将 `token` 转成 `index`的工具类

**功能**

* 由于单个 token 也可能由多个 字母构成，此函数可用来填充
* `get_padding_lengths(self)`
  * 返回 `field` 中各个元素的长度

**一些Field**

* `TextField` : 文本域
* `ListField` : 普通 `Field` 的列表



## Instance

* 多个 `Field` 的集合，一个 `Field` 表示样本的一个字段

**方法**

* `count_vocab_items(self, counter: Dict[str, Dict[str, int]])`: 
  *  会调用 `field` 的 `count_vocab_items(self, counter: Dict[str, Dict[str, int]])` 方法更新 `counter`
     *  统计该 `field` 下的 `tokens`
* `index_fields(self, vocab: Vocabulary)`: 
  * 会调用 `field.index(vocab)` 
  * 通过 `vocab` 将 `token` 转成 `index`
* `get_padding_lengths(self)` 
  *  调用了 `field.get_padding_lengths()`
* `as_tensor_dict(self)` ，返回的字典如下所示，会执行 `padding` 操作
  * 可以从应用中理解这种组织方式
    * 一个 `instance` 会有多个 `field` , `Instance(Dict[str, Field])`
    * 对于同一个 `field` 可能会有不同的 `indexer` 处理, 比如 `token-level, char-level, sub-words-level` . `Field(List[Token], Dict[str, Indexer]])`
    * 不同 `field` 可能会共享 `token->id` 映射 , 比如输入输出都是中文. 这个就由 `indexer` 的 `namespace` 来提供.

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

* 在构建 `vocab` 的时候会调用 `instance.count_vocab_items()` 方法. 会一路调用到 `TokenIndexer`
* 默认的 `padding_idx` 为 `0`
* 默认的 `oov` 为 `1`

```python
# non_padded_namespaces 这个地方来指定哪些 namespace 是不需要 padding 的
def from_instances(cls,
                   instances: Iterable['adi.Instance'],
                   min_count: Dict[str, int] = None,
                   max_vocab_size: Union[int, Dict[str, int]] = None,
                   non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
                   pretrained_files: Optional[Dict[str, str]] = None,
                   only_include_pretrained_words: bool = False,
                   tokens_to_add: Dict[str, List[str]] = None,
                   min_pretrained_embeddings: Dict[str, int] = None) -> 'Vocabulary':
```

* 如何从 `files` 构建 索引:  还是按照行来的.



**构建Vocab的流程**

```python
"""
Instance.count_vocab_items(self, counter: Dict[str, Dict[str, int]])
	遍历执行所有的 Field.count_vocab_items()
		一些Field会遍历执行所有的 Indexer.count_vocab_items()
		LabelField就没有Indexer, 所以就直接在Field级别就 count 了
"""
```



**token->idx的流程**

```python
"""
Instance.index_fields(self, vocab)
	遍历执行所有的 Field.index(vocab)
	一些Field会遍历执行所有的 Indexer.index
	Label 就没有 Indexer, 所以在 Field 级别, 就可以 id 化完成了.
"""
```



**构建Vocab的两种方法**

* `from_instance` :

```python
"""
1. 遍历整个数据集 构建 counter, (确定 indexer 的 begin 和 end 所起的作用)
2. 
"""
```

* `Label` 构建的 `Vocab` 没有 `UNK` 这是为什么呢?

* `from_files` 



## Iterator

* `iterators.PassThroughIterator` : 按照数据集顺序, 一次 `yield` 一个样本
* `iterators.BasicIterator`: 可缓存, 可 `shuffle`
* `iterators.BucketIterator`: 桶迭代
* `iterators.MultiprocessIterator` : 多进程, 可将上述两个包起来



## allennlp 输入流水线的整个流程

```python
"""
fields: 字典, [str]Field

Instance(fields)			 									---> Field(tokens, token_indexers)
	def count_vocab_items(self, counter)	--->   def count_vocab_items(self, counter)
	def index_fields(self, vocab)					--->   def index(self, vocab)


一个field可以有多个token_indexers, 这个比较合适 char-level + word-level
Counter: Dict[str1, Dict[str2, int]], str1为namespace, str2为word/char, int为出现的次数.
该Counter 应该是全局的吧

Field(tokens, token_indexers)           ---> SingleIdTokenIndexer(namespace, )
	def count_vocab_items(self, counter)	---> 	def count_vocab_items(self, token, counter)
	def index(self, vocab)								---> 	def as_padded_tensor .. 这个是干嘛的
																				---> 	def token_to_indice(tokens, vocab, index_name)
	
"""

```

* 写`DatasetReader` 
  * `DatasetReader._read()` 每次 `yield` 一个 `Instance`
* 通过 `DatasetReader` 构建 `Vocabulary`
* 构建 `Iterator`



## TODO

* 如何`padding` , 默认 `0` 为 `padding_idx` 了.如果需要 `padding` 的话, 是否需要 是和 `namespace` 相关的.
* 迭代器如何 `buffer`



## Embedding

* `allennlp.modules.token_embedders.Embedding`
* `allennlp.modules.text_field_embedders.TextFieldEmbedder , BasicTextFieldEmbedder`
  * `TextFieldEmbedder` 是 `TokenEmbdder` 的集合



## 一些特殊符号处理

* `BEGIN, END`:

  * `indexer` 中虽然有 `start_tokens, end_tokens`, 但是执行 `Vocabulary.from_instances` 的时候, 生成的 `Vocabulary` 中并不会包含该 对应的 `tokens` , 所以如果想要加上 `<BEGIN>` 和 `<END>` 的话, 还是需要在构建 `Field` 的时候手动加上 对应的 `Token`. 
  * `indexer` 中的 `start_tokens, end_tokens` 直接忽略就好了.

* `<UNK>, <PAD>` : `Vocabulary` 处理

  * `Vocabulary.from_instances` 会对非`label` `field` 添加 `<UNK>, <PAD>` , 而且在 `save_to_files` 时候会将 `<UNK>` dump 到 文件中
  * `Vocabulary.from_files` 中会自动添加 `<PAD>` 

  