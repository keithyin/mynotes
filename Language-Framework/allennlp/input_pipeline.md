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

* 负责给 `token` 设置 `index` 的工具类
* 二级 `token` 也可以由其完成

**功能**

* 统计 `token` 在语料库中出现的次数，
* 将 `token list` 转成 `token index list`，可以添加开始 `token:<GO>`  和结束 `token:<END>`
* 将 单个 `token list` 填充到期望的 长度。在`mini-batch`化的时候会用到。



## Field

* 用来区分样本中的不同部分（输入域，输出域）
* 构成：`tokens, token_indexers`
  * `tokens` : list of tokens，用来表示一个句子
  * `token_indexers` ：用来负责将 `token` 转成 `index`

**功能**

* 由于单个 token 也可能由多个 字母构成，此函数可用来填充



## Instance

* `Field` 集合，表示一个样本




## Batch

* 用来构建 mini-batch
* 负责填充



