

深度学习训练的代码一般包括以下几个部分：

1. 输入处理 `input_fn`
2. 模型搭建 `model_fn`
3. 模型训练（for循环那一堆）
4. 模型导出 & 参数导出 （有时候可能需要有策略的导出，比如：保留最近几次的模型 etc）

estimator实际就是将 3，4 部分代码帮我们写好了，我们只需要关注 1，2部分的实现就好了。

# 自定义estimator
* tensorflow已经实现了一些常用的 estimator。可以拿来直接用。但是本文更加关注于如何自定义estimator。

* 当我们谈到自定义estimator的时候，我们说的是使用 `tf.estimator.Estimator` 类，而非更加高阶的类，比如:`tf.estimator.DNNClassifier` 
* 自定义estimator的时候：我们需要实现以下几个函数
  * `def input_fn` :   定义 `dataset` 的一系列操作，最终返回 `dataset`
    * `iterator` 的构建在 `Estimator` 处理，不需要我们写代码
  * `def model_fn` : 定义了模型结构 & `train_op`

## input_fn

```python
def input_fn():
  dataset = SomeDataset()
  # parse_function 负责单个样本的解析。
  dataset = dataset.map(lambda record: parse_function(record, is_training))
  dataset = dataset.batch(batch_size)   # 这里不建议使用 padded_batch, 对于pad 操作可以在 model_fn中处理！。
  dataset = dataset.prefetch(FLAGS.prefetch)
  return dataset
```

## model_fn

`model_fn` 的签名如下所示. 

* `model_fn` 负责模型结构搭建 & `train_op`
* 返回一个 `EstimatorSpec`

```python
"""
mode 由 estimator 调用该函数时传入。
	estimator.train 时 mode==tf.estimator.ModeKeys.TRAIN
	estimator.evaluate 时 mode==tf.estimator.ModeKeys.EVAL
	estimator.predict 时 mode==tf.estimator.ModeKeys.PREDICT
"""
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

## train & evaluate

```python
# Create the Estimator，指定 model_fn & model_dir
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

# 控制台打印日志
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

# train one step and display the probabilties
# train 的时候指定 input_fn & hooks
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

* tensorflow 还提供了一个 api，`train & evaluate` 可以一行代码搞定 `train_and_evaluate` . 该API，将 1）训练，2）保存ckpt，3）evaluate，4）导出serving model一起封装了起来，还包括 tensorboard。
  为了使用此API，我们需要提供：
  1. 构建一个 `estimator` 以备使用
  2. 训练：定义好 `TrainSpec`
  3. ckpt & tensorboard：在 `tf.estimator.RunConfig` 中配置好相应参数
  4. 测试：定义好 `EvalSpec`
  5. 导出：`EvalSpec` 里配好 `Exporter`。
     1. 在执行export功能的时候，构建的是一个 **PREDICT** model !所以 model_fn 的PREDICT分支的`EstimatorSpec`中配置好`export_outputs`

`train_and_evaluate` 基本执行流程: 

1. 模型训练，产生 ckpt
2. evaluator 看到有ckpt产生，就开始 evaluate
3. evaluate 之后构建一个 PREDICT 图，将当前 ckpt + grph export 出来。可以用作 serving

```python
tf.estimator.train_and_evaluate(
    estimator, train_spec, eval_spec
)
```



## tensorboard & summary

* 我们在 `model_fn` 中写的 `tf.summary.scalar, tf.summary.hitogram ...` 这个仅仅是 train 时候的 summary。
* `metrics` 那一堆是 `eval` 时候的 summary。
* 如果我们想在eval时候输出除metrics之外的其它 summary 的时候，需要用到 `Hook!`

```python
with tf.name_scope("histogram_summary") as name_scope:
    tf.summary.histogram("pred_cvr", cvr)
    tf.summary.histogram("cvr_cost", cvr_cost)
    
    # output_dir 如此设定是为了和 estimator 那些 metric 的输出保持一致。
    eval_summary_hook = tf.train.SummarySaverHook(save_steps=1,
                                                  output_dir="{}/eval".format(model_ckpt_dir),
                                                  summary_op=tf.summary.merge_all(scope=name_scope))
return tf.estimator.EstimatorSpec(
              mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, export_outputs=None, evaluation_hooks=[summary_hook])
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

* 导出的模型的裸API

```python
estimator.export_savedmodel(export_dir_base, serving_input_receiver_fn,
                            strip_default_attrs=True)
```
This method builds a new graph by first calling the serving_input_receiver_fn() to obtain feature Tensors, and then calling this Estimator's model_fn() to generate the model graph based on those features. It starts a fresh Session, and, by default, restores the most recent checkpoint into it. (A different checkpoint may be passed, if needed.) Finally it creates a time-stamped export directory below the given export_dir_base (i.e., export_dir_base/<timestamp>), and writes a SavedModel into it containing a single MetaGraphDef saved from this Session.



* 当使用`train_and_evaluate API`   时如何进行 模型导出配置. 
  * 构建`Exporter`, 将其传给 `EvalSpec`
  * `tf` 提供了两个 `exporter` 可供使用:  `tf.estimator.FinalExporter, tf.estimator.BestExporter` 
  * `tf.estimator.FinalExporter` : 导出最近的 `ckpt`
  * `tf.estimator.BestExporter` : 导出loss低的 `ckpt` （较为常用）

```python
tf.estimator.BestExporter(
    name='best_exporter', 
  	serving_input_receiver_fn=None,
    event_file_pattern='eval/*.tfevents.*', #为了抢占安全，一定要指定。抢占安全？
  	compare_fn=_loss_smaller,
    assets_extra=None,  # 除了导出模型，还需要导出的一些文件，比如：vocab文件？
  	as_text=False, 
  	exports_to_keep=5
)
```

* 关于 `compare_fn`: a function that compares two evaluation results and returns true if current evaluation result is better. Follows the signature: `def compare_fn(best_eval_result, current_eval_result) -> bool`
  * `best_eval_result, current_eval_result` 实际是 `estimator.evaluate` 的返回值，是个 `dict` . 
  * `dict` 的 `key` 包含：1）我们定义的 `eval_metrics` 中的那些 `key` ，2）`loss` , 该 `loss` 代表`EstimatorSpec` 中传入的 `loss`, `estimator.evaluator` 返回的 `{"loss": 均值}` 。




### 分布式训练

* 代码部分无需修改，配置好环境变量 `TF_CONFIG` 即可：`TF_CONFIG`是个json string。
* https://github.com/tensorflow/docs/blob/r1.15/site/en/api_docs/python/tf/estimator/train_and_evaluate.md

```
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "chief": ["host0:port"],
        "worker": ["host1:port", "host2:port", "host3:port"],
        "ps": ["host4:port", "host5:port"]
    },
   "task": {"type": "worker", "index": 0} // type：当前进程角色🎭，index：当前进程在上述列表中的索引。
})
```

### Serving
TODO

# 涉及到的Config总结

* tf.ConfigProto: 用来配置 session 资源
  * https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/ConfigProto.
```python
ConfigProto
allow_soft_placement	bool allow_soft_placement
cluster_def	ClusterDef cluster_def
device_count	repeated DeviceCountEntry device_count
device_filters	repeated string device_filters
experimental	Experimental experimental
gpu_options	GPUOptions gpu_options
graph_options	GraphOptions graph_options
inter_op_parallelism_threads	int32 inter_op_parallelism_threads
intra_op_parallelism_threads	int32 intra_op_parallelism_threads
isolate_session_state	bool isolate_session_state
log_device_placement	bool log_device_placement
operation_timeout_in_ms	int64 operation_timeout_in_ms
placement_period	int32 placement_period
rpc_options	RPCOptions rpc_options
session_inter_op_thread_pool	repeated ThreadPoolOptionProto session_inter_op_thread_pool
use_per_session_threads	bool use_per_session_threads
```

* tf.estimator.RunConfig: estimator 的运行Config，包含 `ConfigProto`，同时也有一些其它estimator相关的配置
  * checkpoint 配置，
  * summary 配置
  * 分布式训练配置？
```python

class RunConfig(object):
  """This class specifies the configurations for an `Estimator` run."""
  def __init__(self,
               model_dir=None,    # ckpt 保存位置
               tf_random_seed=None,
               save_summary_steps=100,  # 保存 summary 的 interval
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



 `**Spec` 命名的类，是为了封装 函数的输入而存在的。。。
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



# 模型的导出与导入
https://github.com/tensorflow/docs/blob/r1.15/site/en/guide/saved_model.md



# estimator导出的模型如何load

> estimator导出的model不需要用 tensorflow-serving 时怎么办
* 使用 `saved_model_cli show --dir exported_model_dir --all` 先检查一下导出来的图，看一下模型的 输入 & 输出都是什么玩意。
* 然后就可以撸代码了

https://github.com/keithyin/mynotes/blob/master/Language-Framework/tensorflow/high-level-api/load_model_saved_by_estimator.py



# 注意⚠️

* global_step在evaluate时候是不会累加的。这也是非常合理的。



# 使用`train_and_evaluate`整体代码

```python
from tensorflow.saved_model import signature_constants
import tensorflow as tf

def train_input_fn():
  return dataset

def eval_input_fn():
  return dataset

def serving_input_receiver_fn():
  serialized_tf_examples = tf.placeholder(shape=[None], dtype=tf.string)
  
  # 请求 tf-serving 时传的 数据。
  receiver_tensor = {'examples': serialized_tf_examples}
  features = tf.parse_example(serialized_tf_examples, feature_description)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)

def model_fn(features, labels, mode):
  predicted_vals = net(features)
  
  """
  当estimator export模型的时候，mode 会传入 tf.estimator.ModeKeys.PREDICT
  此时会走该分支
  """
  if mode == tf.estimator.ModeKeys.PREDICT:
    export_outputs = {
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput({
                "cvr": tf.squeeze(cvr_prob, axis=1, name="cvr"),
                
                "user_id": tf.identity(features['user_id'], name='user_id')
            })
        }
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
  
  loss = compute_loss(predicted_vals, labels)
  
  if mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  	}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
  
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
  train_op = optimizer.minimize(
    loss=loss,
    global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  
 

```

> 先判断 `mode == tf.estimator.ModeKeys.PREDICT`, 再判断 `mode == tf.estimator.ModeKeys.EVAL` 最终是 `TRAIN` 的原因是，`predict` 仅需要 预估过程即可，不需 loss计算。同理：`EVAL` 不需要 梯度计算。

```python
session_config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False,
                                operation_timeout_in_ms=0,
                                device_filters=device_filters) # device_filters不知干啥用的。。

run_config = tf.estimator.RunConfig(
        model_dir=output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        session_config=session_config,
        log_step_count_steps=10
    )

estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config
    )

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps, hooks=None)

best_exporter = tf.estimator.BestExporter(serving_input_receiver_fn=serving_input_receiver_fn)
eval_spec = tf.estimator.EvalSpec(
    eval_input_fn, steps=100, name=None, hooks=None, exporters=[best_exporter],
    start_delay_secs=120, throttle_secs=600
)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
```





# 如何解决train过程auc累积问题

* 解决方案：将 metric 的的中间结果清零 🆑。代码如下
```python
with tf.variable_scope("train_metrics_scope"):
    metric_auc = tf.metrics.auc(true_cvr, cvr, num_thresholds=10240)
    metric_cvr_loss = tf.metrics.mean(cvr_loss)
    metric_cost_loss = tf.metrics.mean_squared_error(true_cost, cvr_cost, weights=true_cvr)

    is_metric_reset_step = tf.equal(global_step % 10000, 0)
    metric_reset_op = tf.cond(is_metric_reset_step,
			      lambda: tf.group([tf.assign(ref, tf.zeros_like(ref))
						for ref in tf.local_variables() if
						'train_metrics_scope' in ref.op.name]),
			      lambda: tf.no_op())

with tf.name_scope("train_metrics_summary"):
    tf.summary.scalar("auc", metric_auc[0])
    tf.summary.scalar("cvr_loss", metric_cvr_loss[0])
    tf.summary.scalar("cost_loss", metric_cost_loss[0])

# ...

train_op = tf.train.AdamOptimizer(
            learning_rate=lr).minimize(tot_loss,
				       global_step=global_step)
train_op = tf.group(train_op, metric_reset_op, metric_auc[1], metric_cvr_loss[1], metric_cost_loss[1])
```



# 参考资料

https://github.com/tensorflow/docs/blob/r1.15/site/en/guide/custom_estimators.md
https://github.com/tensorflow/docs/blob/r1.15/site/en/tutorials/estimators/cnn.ipynb
https://github.com/tensorflow/docs/blob/r1.13/site/en/guide/saved_model.md

https://github.com/tensorflow/docs/blob/r1.15/site/en/api_docs/python/tf/estimator/train_and_evaluate.md

