# 构建多GPU代码

## 结构
1. 先构建单GPU代码
2. 写个函数`multi_gpu_model(num_gpus)`来生成多`GPU`代码，并将对象保存在`collection`中
3. feed data
4. run

## 如何构建单GPU代码
之前文章已经介绍过。
不要在单GPU代码中创建`optimizer op`,因为是`multi gpu`，所以参数更新的操作是所有的`GPU`计算完梯度之后，才进行更新的。

## 如何实现`multi_gpu_model`函数

```python
def multi_gpu_model(num_gpus=1):
  grads = []
  for i in range(num_gpus):
    with tf.device("/gpu:%d"%i):
      with tf.name_scope("tower_%d"%i):
        model = Model(is_training, config, scope)
        # 放到collection中，方便feed的时候取
        tf.add_to_collection("train_model", model)
        grads.append(model.grad) #grad 是通过tf.gradients(loss, vars)求得
        #以下这些add_to_collection可以直接在模型内部完成。
        # 将loss放到 collection中， 方便以后操作
        tf.add_to_collection("loss",model.loss)
        #将predict放到collection中，方便操作
        tf.add_to_collection("predict", model.predict)
        #将 summary.merge op放到collection中，方便操作
        tf.add_to_collection("merge_summary", model.merge_summary)
        # ...
  with tf.device("cpu:0"):
    averaged_gradients = average_gradients(grads)# average_gradients后面说明
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    train_op=opt.apply_gradients(zip(average_gradients,tf.trainable_variables()))

  return train_op
```

## 如何`feed data`
```python

def generate_feed_dic(model, feed_dict, batch_generator):
  x, y = batch_generator.next_batch()
  feed_dict[model.x] = x
  feed_dict[model.y] = y

```
## 如何实现run_epoch

```python
#这里的scope是用来区别 train 还是 test
def run_epoch(session, data_set, scope, train_op=None, is_training=True):
  batch_generator = BatchGenerator(data_set, batch_size)
  ...
  ...
  if is_training and train_op is not None:
    models = tf.get_collection("train_model")
    # 生成 feed_dict
    feed_dic = {}
    for model in models:
      generate_feed_dic(model, feed_dic, batch_generator)
    #生成fetch_dict
    losses = tf.get_collection("loss", scope)#保证了在 test的时候，不会fetch train的loss
    ...
    ...

```

## main函数
main 函数干了以下几件事：
1. 数据处理
2. 建立多GPU训练模型
3. 建立单/多GPU测试模型
4. 创建`Saver`对象和`FileWriter`对象
5. 创建`session`
6. run_epoch
```python
data_process()
with tf.name_scope("train") as train_scope:
  train_op = multi_gpu_model(..)
with tf.name_scope("test") as test_scope:
  model = Model(...)
saver = tf.train.Saver()
# 建图完毕，开始执行运算
with tf.Session() as sess:
  writer = tf.summary.FileWriter(...)
  ...
  run_epoch(...,train_scope)
  run_epoch(...,test_scope)
```

## 如何编写average_gradients函数
```python
def average_gradients(grads):#grads:[[grad0, grad1,..], [grad0,grad1,..]..]
  averaged_grads = []
  for grads_per_var in zip(*grads):
    grads = []
    for grad in grads_per_var:
      expanded_grad = tf.expanded_dim(grad,0)
      grads.append(expanded_grad)
    grads = tf.concat_v2(grads, 0)
    grads = tf.reduce_mean(grads, 0)
    averaged_grads.append(grads)

  return averaged_grads
```
还有一个版本，但是不work，不知为啥
```python
def average_gradients(grads):#grads:[[grad0, grad1,..], [grad0,grad1,..]..]
  averaged_grads = []
  for grads_per_var in zip(*grads):
    grads = tf.reduce_mean(grads_per_var, 0)
    averaged_grads.append(grads)
  return averaged_grads
```
