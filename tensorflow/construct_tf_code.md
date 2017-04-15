# 如何构建TF代码

`batch_size`: batch的大小
`mini_batch`: 将训练样本以`batch_size`分组
`epoch_size`: 样本分为几个`min_batch`
`num_epoch` : 训练几轮
## 读代码的时候应该关注的几部分
1. 如何处理数据
2. 如何构建计算图
3. 如何计算梯度
4. 如何`Summary`，如何`save`模型参数
5. 如何执行计算图


## 写一个将数据分成训练集,验证集和测试集的函数
```python
train_set, valid_set, test_set = split_set(data)
```

## 最好写一个管理数据的对象，将原始数据转化成`mini_batch`
```python
class DataManager(object):
  #raw_data为train_set, valid_data或test_set
  def __init__(self, raw_data, batch_size):
    self.raw_data = raw_data
    self.batch_size = batch_size
    self.epoch_size = len(raw_data)/batch_size
    self.counter = 0 #监测batch index
  def next_batch(self):
    ...
    self.counter += 1
    return batched_x, batched_label, ...
```

## `计算图`的构建在`Model`类中的`__init__()`中完成,并设置is_training参数
**优点：**
1. 因为如果我们在训练的时候加`dropout`的话，那么在测试的时候是需要把这个`dropout`层去掉的。这样的话，在写代码的时候，你就可以创建两个对象。这就相当于建了两个`模型`，然后让这两个`模型`参数共享，就可以达到`训练`和`测试`一起运行的效果了。具体看下面代码。
```python
class Model(object):
  def __init__(self, is_training, config, scope,...):#scope可以使你正确的summary
    self.is_training = is_training
    self.config = config
    #placeholder:用于feed数据
    # 一个train op
    self.graph(self.is_training) #构建图
    self.merge_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,scope))
  def graph(self,is_training):
    ...
    #定义计算图
    self.predict = ...
    self.loss = ...
```
## 写个`run_epoch`函数
`batch_size`: batch的大小
`mini_batch`: 将训练样本以`batch_size`分组
`epoch_size`: 样本分为几个`min_batch`
`num_epoch` : 训练几轮
**如何编写run_epoch函数**
```python
#eval_op是用来指定是否需要训练模型，需要的话，传入模型的eval_op
#draw_ata用于接收 train_data,valid_data或test_data
def run_epoch(raw_data ，session, model, is_training_set, ...):
  data_manager = DataManager(raw_data, model.config.batch_size)

  #通过is_training_set来决定fetch哪些Tensor
  #add_summary, saver.save(....)
```

## 如何组织main函数
1. 分解原始数据为train，valid，test
2. 设置默认图
3. 建图 trian, test 分别建图
4. 一个或多个`Saver`对象，用来保存模型参数
5. 创建session， 初始化变量
6. 一个`summary.FileWriter`对象，用来将`summary`写入到硬盘中
7. run epoch

`FileWriter` 和 `Saver`对象，一个计算图只需要一个就够了，所以放在Model类的外面
## 附录
本篇博文总结下面代码写成， 有些地方和源码之间有不同。
下面是截取自官方代码：
```python
class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        size, forget_bias=0.0, state_is_tuple=True)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell] * config.num_layers, state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.nn.rnn(cell, inputs,
    #                            initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat_v2(outputs, 1), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(input_.targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)

def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.contrib.deprecated.scalar_summary("Training Loss", m.cost)
      tf.contrib.deprecated.scalar_summary("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.contrib.deprecated.scalar_summary("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      test_perplexity = run_epoch(session, mtest)
      print("Test Perplexity: %.3f" % test_perplexity)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()
```

**参考资料**
源码地址[https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py](https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py)
