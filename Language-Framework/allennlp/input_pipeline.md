# input pipeline

* tokenization
* pad
* neumericalization
* vocab




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

