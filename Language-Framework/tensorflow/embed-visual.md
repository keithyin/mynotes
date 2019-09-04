# embedding visualization

1. 从 `checkpoint` 中读取 `embedding`即：`checkpoint`中的任何`2D Tensor`
2. 如何将 `点` 和 `label` 配对。

## 如何做
1.
```python
#Setup a 2D tensor that holds your embedding(s)
embedding_var = tf.Variable(....)
```
2.
```python
# 创建一个saver，用于save你的模型
saver = tf.train.Saver()
saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), step)
```
## 如何将embedding与真实数据联系起来
```python
from tensorflow.contrib.tensorboard.plugins import projector

# Create randomly initialized embedding weights which will be trained.
N = 10000 # Number of items (vocab size).
D = 200 # Dimensionality of the embedding.
embedding_var = tf.Variable(tf.random_normal([N,D]), name='word_embedding')

# Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter(LOG_DIR)

# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# read this file during startup.
projector.visualize_embeddings(summary_writer, config)
```

## 缺点
为什么不能 可视化`Tensor`

## 参考资料
https://www.tensorflow.org/guide/embedding
