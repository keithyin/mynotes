# 诡异的现象
version: tensorflow1.15, python3.6

`feature_column.shared_embedding_columns` & `feature_column.sequence_numeric_column` 使用 `feature_column.make_parse_example_spec`会报错。错在 `isinstance(col, _FeatureColumn)`上。

`feature_column.shared_embedding_columns` & `feature_column.sequence_numeric_column` 使用 `feature_column.input_layer`会报错。错在 `isinstance(col, _FeatureColumn)`上。



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

tf.feature_column.shared_embedding_columns(
    categorical_columns, dimension, combiner='mean', initializer=None,
    shared_embedding_collection_name=None, ckpt_to_load_from=None,
    tensor_name_in_ckpt=None, max_norm=None, trainable=True
)
```

