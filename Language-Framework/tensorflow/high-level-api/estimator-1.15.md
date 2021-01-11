深度学习训练的代码一般包括以下几个部分：
1. 输入处理 `input_fn`
2. 模型搭建 `model_fn`
3. 模型训练（for循环那一堆）
4. 模型导出 & 参数导出 （有时候可能需要有策略的导出，比如：保留最近几次的模型 etc）

estimator实际就是将 3，4部分代码帮我们写好了，我们只需要关注 1，2部分的实现就好了。

# estimator
tensorflow已经实现了一些常用的 estimator。
* Estimator
* ...

# input
```python
def input_fn():
  return features, labels # 只需要返回这两个玩意就可以了。features, labels, 可以是 Tensor，可以是 list of tensor，可以是 string to tensor dict
```

# model
`model_fn` 的签名如下所示.
```python
def model_fn(features, labels, mode):
  ...
  ...
  return tf.estimator.EstimatorSpec()
```
```python
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
```

# estimator
```python
# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

# train one step and display the probabilties
mnist_classifier.train(
    input_fn=train_input_fn,
    steps=1, # 训练几个step。
    hooks=[logging_hook])
    
# evaluate model！
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
```

```python
# 该api挺好用。此函数 运行 train 的时候，每导出一次 ckpt, 都会跑一遍 eval。至于这两个操作是不是并行就不清楚了。
tf.estimator.train_and_evaluate(
    estimator, train_spec, eval_spec
)
```

# 模型的导出
> 模型导出并不是将训练时候的Graph直接导出，而是新建一个Graph，然后再导出。

下面介绍，Estimator训练好的模型如何导出以做它用。
对于导出模型主要关注的点是，模型的输入是什么，输出是什么。

### 构建Serving Input：
主要是在做 输入预处理部分
* 添加 placeholder：（提供Graph显式入口）
* 对输入进行处理
```python
feature_spec = {'foo': tf.FixedLenFeature(...),
                'bar': tf.VarLenFeature(...)}

def serving_input_receiver_fn():
  """An input receiver that expects a serialized tf.Example."""
  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[default_batch_size],
                                         name='input_example_tensor')
  receiver_tensors = {'examples': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors) # 将placeholder 和 模型输入 tensor 打包起来。
```
https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/io/parse_example

### 指定模型的输出: 
输出在 `model_fn` 返回的 `EstimatorSpec` 中指定。每个导出的输出都需要是一个`ExportOutput`子类的对象，例如 `tf.estimator.export.ClassificationOutput`, `tf.estimator.export.RegressionOutput`, or `tf.estimator.export.PredictOutput`.
```python
def model_fn(...):
  logit = model(input)
  exported = {
    "prediction": tf.estimator.export.PredictOutput(logit)
  }
  return tf.extimator.EstimatorSpec(export_outputs=exported, ...)
```
### 导出模型
```python
estimator.export_savedmodel(export_dir_base, serving_input_receiver_fn,
                            strip_default_attrs=True)
```
This method builds a new graph by first calling the serving_input_receiver_fn() to obtain feature Tensors, and then calling this Estimator's model_fn() to generate the model graph based on those features. It starts a fresh Session, and, by default, restores the most recent checkpoint into it. (A different checkpoint may be passed, if needed.) Finally it creates a time-stamped export directory below the given export_dir_base (i.e., export_dir_base/<timestamp>), and writes a SavedModel into it containing a single MetaGraphDef saved from this Session.
  
### Serving
TODO

# 涉及到的Config总结

* tf.ConfigProto: 用来配置 session 资源
  * https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/ConfigProto.
  * 
* tf.estimator.RunConfig: estimator 的运行Config，包含 `ConfigProto`，同时也有一些其它estimator相关的配置
  * checkpoint 配置，summary 配置。
```python


class RunConfig(object):
  """This class specifies the configurations for an `Estimator` run."""
  def __init__(self,
               model_dir=None,
               tf_random_seed=None,
               save_summary_steps=100,
               save_checkpoints_steps=_USE_DEFAULT,
               save_checkpoints_secs=_USE_DEFAULT,
               session_config=None, # tf.ConfigProto
               keep_checkpoint_max=5,
               keep_checkpoint_every_n_hours=10000,
               log_step_count_steps=100,
               train_distribute=None,
               device_fn=None,
               protocol=None,
               eval_distribute=None,
               experimental_distribute=None,
               experimental_max_worker_delay_secs=None,
               session_creation_timeout_secs=7200):
```

感觉 `**Spec` 命名的类，是为了封装 函数的输入而存在的。。。
* tf.estimator.EstimatorSpec: `model_fn` 返回的结构体。用来指明 `model` 的一些基本信息
* tf.estimator.TrainSpec: 训练 model 需要的一些参数
* tf.estimator.EvalSpec: 评估时候 需要的一些参数

```python
class TrainSpec(
    collections.namedtuple('TrainSpec', ['input_fn', 'max_steps', 'hooks'])):
    
class EvalSpec(
    collections.namedtuple('EvalSpec', [
        'input_fn', 'steps', 'name', 'hooks', 'exporters', 'start_delay_secs',
        'throttle_secs'
    ])):
    """
    exporters: Iterable of `Exporter`s, or a single one, or `None`.
        `exporters` will be invoked after each evaluation.
    """

```


# 参考资料
https://github.com/tensorflow/docs/blob/r1.15/site/en/guide/custom_estimators.md
https://github.com/tensorflow/docs/blob/r1.15/site/en/tutorials/estimators/cnn.ipynb
https://github.com/tensorflow/docs/blob/r1.13/site/en/guide/saved_model.md
