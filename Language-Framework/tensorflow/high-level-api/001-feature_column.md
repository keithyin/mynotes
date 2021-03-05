# 诡异的现象
version: tensorflow1.15, python3.6



> Feature_column: 是对 parse_example 之后的特征进行处理！！！
>
> feature_column 可以产生 parse_example 所需的 feature_description



`feature_column` 的 `api` 可以简单的看作两层？

* 第一层：离tfrecord最近的一层。这层上的 `api` 基本都会有参数 `key` ，用来表示 `tfrecord feature key`
  * 这些 `feature_column` 不仅能够帮我们简单处理特征，还可以帮我们生成 `feature_description`
  * 第一层的特征的功能有：
    * 数值特征透传
    * string特征id化
    * 离散特征id化
    * cross-feature：估计这个和 数值特征没法一起开心玩耍了. cross-feature 也可以出现在第二层

```python
# 单值特征 ---------------

tf.feature_column.categorical_column_with_hash_bucket(
    key, hash_bucket_size, dtype=tf.dtypes.string
)

# 当特征值是 整数，且希望 特征值本身为 categorical ID 时，使用该方法
# default_value 必须在 [0, num_buckets) 区间内。
tf.feature_column.categorical_column_with_identity(
    key, num_buckets, default_value=None
)


tf.feature_column.categorical_column_with_vocabulary_file(
    key, vocabulary_file, vocabulary_size=None, num_oov_buckets=0,
    default_value=None, dtype=tf.dtypes.string
)
tf.feature_column.categorical_column_with_vocabulary_list(
    key, vocabulary_list, dtype=None, default_value=-1, num_oov_buckets=0
)

tf.feature_column.crossed_column(
    keys, hash_bucket_size, hash_key=None
)
tf.feature_column.numeric_column(
    key, shape=(1,), default_value=None, dtype=tf.dtypes.float32, normalizer_fn=None
)

# 序列特征 ------------------
tf.feature_column.sequence_categorical_column_with_hash_bucket(
    key, hash_bucket_size, dtype=tf.dtypes.string
)
tf.feature_column.sequence_categorical_column_with_identity(
    key, num_buckets, default_value=None
)
tf.feature_column.sequence_categorical_column_with_vocabulary_file(
    key, vocabulary_file, vocabulary_size=None, num_oov_buckets=0,
    default_value=None, dtype=tf.dtypes.string
)
tf.feature_column.sequence_categorical_column_with_vocabulary_list(
    key, vocabulary_list, dtype=None, default_value=-1, num_oov_buckets=0
)
tf.feature_column.sequence_numeric_column(
    key, shape=(1,), default_value=0.0, dtype=tf.dtypes.float32, normalizer_fn=None
)
```

* 第二层：这层是特征的进一步处理。
  * bucketing
  * cross-feature。因为 cross-feature 的keys 可以是 cat-column. 

```python
# 连续特征离散化
tf.feature_column.bucketized_column(
    source_column, boundaries
)

tf.feature_column.crossed_column(
    keys, hash_bucket_size, hash_key=None
)
```

* 第三层：id向量化
  * id->onehot
  * Id->denseEmb
  * 需要注意的一点就是：这里的 indicator & embedding 仅支持二维输入。shape=[b, num_fea]。如果 num_fea > 1：indicator 就变成了 multi-hot。embedding 也会根据 combiner 对 多个特征进行处理。

```python
# id -> onehot
tf.feature_column.indicator_column(
    categorical_column
)
# id -> embedding
tf.feature_column.embedding_column(
    categorical_column, dimension, combiner='mean', initializer=None,
    ckpt_to_load_from=None, tensor_name_in_ckpt=None, max_norm=None, trainable=True
)

# 该方法会返回 list，注意处理一下！！
tf.feature_column.shared_embedding_columns(
    categorical_columns, dimension, combiner='mean', initializer=None,
    shared_embedding_collection_name=None, ckpt_to_load_from=None,
    tensor_name_in_ckpt=None, max_norm=None, trainable=True
)
```



# 两个重要方法

## tf.feature_column.make_parse_example_spec

* 生成 `feature_description` 的工具方法

```python
# Define features and transformations
feature_a = categorical_column_with_vocabulary_file(...)
feature_b = numeric_column(...)
feature_c_bucketized = bucketized_column(numeric_column("feature_c"), ...)
feature_a_x_feature_c = crossed_column(
    columns=["feature_a", feature_c_bucketized], ...)

feature_columns = set(
    [feature_b, feature_c_bucketized, feature_a_x_feature_c])

features = tf.io.parse_example(
    serialized=serialized_examples,
    features=make_parse_example_spec(feature_columns))

```



## tf.feature_column.input_layer

* `tf.feature_column.*_column*` 各种方法仅仅是声明 对于特征的操作，还没有真正 作用到特征上去。
* `input_layer` 负责对 执行对特征定义的一系列操作。

```python
price = numeric_column('price')
keywords_embedded = embedding_column(
    categorical_column_with_hash_bucket("keywords", 10K), dimensions=16)
columns = [price, keywords_embedded, ...]
features = tf.io.parse_example(..., features=make_parse_example_spec(columns))

# features: 从 tfrecord 解析出来的 batch feature，columns：在这些特征上定义的转换操作。
# 最终，该函数返回一个2-D tensor, [b, processed_fea_size]
dense_tensor = input_layer(features, columns)
```



# 一些限制

* `feature_column.sequence_numeric_column`  不能出现在 ``feature_column.make_parse_example_spec`` 和 `feature_column.input_layer`  :
  * 不能用在 `input_layer`还是比较合理的，因为 `sequence_numeric_column` 就是直接扔到模型中的。并不需要其它操作。
  * 不能用在 `make_parse_example_spec` 是为啥。。。
* `bucketized_column()` 的 `source_column` 必须是 `numeric_column()` 
  * 为啥 `source_column` 不能是 `sequence_numeric_column()` 呢？



