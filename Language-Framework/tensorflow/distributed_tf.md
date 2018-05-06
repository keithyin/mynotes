# 分布式Tensorflow
最近在学习怎么分布式Tensorflow训练深度学习模型,看官网教程看的云里雾里,最终结合着其它资料,终于对分布式Tensorflow有了些初步了解.
## gRPC (google remote procedure call)
分布式Tensorflow底层的通信是`gRPC`
gRPC首先是一个RPC,即`远程过程调用`,通俗的解释是:假设你在本机上执行一段代码`num=add(a,b)`,它调用了一个过程 `call`,然后返回了一个值`num`,你感觉这段代码只是在本机上执行的, 但实际情况是,本机上的`add`方法是将参数打包发送给服务器,然后服务器运行服务器端的`add`方法,返回的结果再将数据打包返回给客户端.

## Cluster.Job.Task
Job是Task的集合.
Cluster是Job的集合

**为什么要分成Cluster,Job,和Task呢?**

首先,我们介绍一下Task:Task就是主机上的一个进程,在大多数情况下,一个机器上只运行一个Task.

为什么`Job`是`Task`的集合呢?   在分布式深度学习框架中,我们一般把`Job`划分为`Parameter`和`Worker`,`Parameter Job`是管理参数的存储和更新工作.`Worker Job`是来运行`ops`.如果参数的数量太大,一台机器处理不了,这就要需要多个`Tasks`.

`Cluster` 是 `Jobs` 的集合: `Cluster`(集群),就是我们用的集群系统了

## 如何创建集群
从上面的描述我们可以知道,组成`Cluster`的基本单位是`Task`(动态上理解,主机上的一个进程,从静态的角度理解,`Task`就是我们写的代码).我们只需编写`Task`代码,然后将代码运行在不同的主机上,这样就构成了`Cluster`(集群)

### 如何编写`Task`代码
首先,`Task`需要知道集群上都有哪些主机,以及它们都监听什么端口.`tf.train.ClusterSpec()`就是用来描述这个.
```python
tf.train.ClusterSpec({
    "worker": [
        "worker_task0.example.com:2222",# /job:worker/task:0 运行的主机
        "worker_task1.example.com:2222",# /job:worker/task:1 运行的主机
        "worker_task2.example.com:2222"# /job:worker/task:3 运行的主机
    ],
    "ps": [
        "ps_task0.example.com:2222",  # /job:ps/task:0 运行的主机
        "ps_task1.example.com:2222"   # /job:ps/task:0 运行的主机
    ]})
```
这个`ClusterSec`告诉我们,我们这个`Cluster`(集群)有两个`Job`(worker.ps),`worker`中有三个`Task`(即,有三个`Task`执行`Tensorflow op`操作)

然后,将`ClusterSpec`当作参数传入到 `tf.train.Server()`中,同时指定此`Task`的`Job_name`和`task_index`.
```python
#jobName和taskIndex是函数运行时,通过命令行传递的参数
server = tf.train.Server(cluster, job_name=jobName, task_index=taskIndex)
```

下面代码描述的是,一个`cluster`中有一个`Job,`叫做(`worker`), 这个`job`有两个`task`,这两个`task`是运行在两个主机上的
```python
#在主机(10.1.1.1)上,实际是运行以下代码
cluster = tf.train.ClusterSpec({"worker": ["10.1.1.1:2222", "10.1.1.2:3333"]})
server = tf.train.Server(cluster, job_name="local", task_index=0)

#在主机(10.1.1.2)上,实际运行以下代码
cluster = tf.train.ClusterSpec({"worker": ["10.1.1.1:2222", "10.1.1.2:3333"]})
server = tf.train.Server(cluster, job_name="local", task_index=1)
```
`tf.trian.Server`干了些什么呢?
首先,一个`tf.train.Server`包含了: 本地设备(GPUs,CPUs)的集合,可以连接到到其它`task`的`ip:port`(存储在`cluster`中), 还有一个`session target`用来执行分布操作.还有最重要的一点就是,它创建了一个服务器,监听`port`端口,如果有数据传过来,他就会在本地执行(启动`session target`,调用本地设备执行运算),然后结果返回给调用者.

我们继续来写我们的`task`代码:在你的`model`中指定分布式设备
```python
with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(...)
  biases_1 = tf.Variable(...)

with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(...)
  biases_2 = tf.Variable(...)

with tf.device("/job:worker/task:0"): #映射到主机(10.1.1.1)上去执行
  input, labels = ...
  layer_1 = tf.nn.relu(tf.matmul(input, weights_1) + biases_1)
  logits = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
with tf.device("/job:worker/task:1"): #映射到主机(10.1.1.2)上去执行
  input, labels = ...
  layer_1 = tf.nn.relu(tf.matmul(input, weights_1) + biases_1)
  logits = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
  # ...
  train_op = ...
with tf.Session("grpc://10.1.1.2:3333") as sess:#在主机(10.1.1.2)上执行run
  for _ in range(10000):
    sess.run(train_op)
```
这里我还有一个疑问:为什么使用了`tf.device("/job:worker/task:1")`,还要用`tf.Session("grpc://10.1.1.2:3333")`,直接用`tf.Session()`为什么就不行?
我感觉这是和gRPC有关.`客户存根`需要服务器地址才能向服务器发送数据.
`with tf.Session("grpc://..")`是指定`gprc://..`为`master`,由`master`将`op`分发给对应的`task`

**写分布式程序时,我们需要关注一下问题:**
(1) 使用`In-graph replication`还是`Between-graph replication`

`In-graph replication`:一个`client`(显示调用tf::Session的进程),将里面的`参数`和`ops`指定给对应的`job`去完成.数据分发只由一个`client`完成.

`Between-graph replication`:下面的代码就是这种形式,有很多独立的`client`,各个`client`构建了相同的`graph`(包含参数,通过使用`tf.train.replica_device_setter`,将这些参数映射到`ps_server`上.)

(2)`同步训练`,还是`异步训练`

`Asynchronous training`:一个有`N-replicas`的异步训练,每个`replica`训练完,都会独立的`apply` `gradients` to `variables`.顺序取决与`replica`的训练速度.这可能导致一个`replica`计算的`gradients`是基于`m`步之前的`variable`.

`Synchronous training`:在这种方式中,每个`graph`的副本读取相同的`parameter`的值,并行的计算`gradients`,然后将所有计算完的`gradients`放在一起处理.`Tensorlfow`提供了函数(`tf.train.SyncReplicasOptimizer`)来处理这个问题(在`Between-graph replication`情况下),在`In-graph replication`将所有的`gradients`平均就可以了

(3) `Between-graph` `Synchoronous`
编写`Between-graph`的同步训练的代码的时,最主要的是通过`tf.trian.SyncReplicasOptimizer`完成同步工作,这种方式通过将`N-replicas`计算后的所有`gradients`收集起来,然后加起来,再一次性把处理后的`gradients`给`variable`.
使用`tf.trian.SyncReplicasOptimizer`,需要一个`chief worker`,执行变量初始化,将`N-replicas`的`gradients`收集起来处理,将处理后的`gradients`给`variable` 一系列工作.

```python
# Create any optimizer to update the variables, say a simple SGD:
opt = GradientDescentOptimizer(learning_rate=0.1)

# Wrap the optimizer with sync_replicas_optimizer with 50 replicas: at each
# step the optimizer collects 50 gradients before applying to variables.
opt = tf.SyncReplicasOptimizer(opt, replicas_to_aggregate=50,
          replica_id=task_id, total_num_replicas=50)
# Note that if you want to have 2 backup replicas, you can change
# total_num_replicas=52 and make sure this number matches how many physical
# replicas you started in your job.

# Some models have startup_delays to help stabilize the model but when using
# sync_replicas training, set it to 0.

# Now you can call `minimize()` or `compute_gradients()` and
# `apply_gradients()` normally
grads = opt.minimize(total_loss, global_step=self.global_step)


# You can now call get_init_tokens_op() and get_chief_queue_runner().
# Note that get_init_tokens_op() must be called before creating session
# because it modifies the graph.
init_token_op = opt.get_init_tokens_op()
chief_queue_runner = opt.get_chief_queue_runner()
sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir=train_dir,
                             init_op=init_op,
                             recovery_wait_secs=1,
                             global_step=global_step)
sess = sv.prepare_or_wait_for_session(FLAGS.worker_grpc_url,
                                          config=sess_config)
# After the session is created by the Supervisor and before the main while
# loop:
if is_chief and FLAGS.sync_replicas:
  sv.start_queue_runners(sess, [chief_queue_runner])
  # Insert initial tokens to the queue.
  sess.run(init_token_op)

```
一个完整的例子(异步),来自官网[链接](https://www.tensorflow.org/versions/r0.10/how_tos/distributed/index.html#distributed-tensorflow):
```python
#由于我们是相同的代码运行在不同的主机上
import tensorflow as tf

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS
```

由于是相同的代码运行在不同的主机上,所以要传入`job_name`和`task_index`加以区分,而`ps_hosts`和`worker_hosts`对于所有主机来说,都是一样的,用来描述集群的

```python
def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
```

我们都知道,服务器进程如果执行完的话,服务器就会关闭.为了是我们的`ps_server`能够一直处于监听状态,我们需要使用`server.join()`.这时,进程就会`block`在这里.至于为什么`ps_server`刚创建就`join`呢:原因是因为下面的代码会将`参数`指定给`ps_server`保管,所以`ps_server`静静的监听就好了.

```python
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):
```

`tf.train.replica_device_setter(ps_tasks=0, ps_device='/job:ps', worker_device='/job:worker', merge_devices=True, cluster=None, ps_ops=None))`,返回值可以被`tf.device`使用,指明下面代码中`variable`和`ops`放置的设备.

example:
```python
# To build a cluster with two ps jobs on hosts ps0 and ps1, and 3 worker
# jobs on hosts worker0, worker1 and worker2.
cluster_spec = {
    "ps": ["ps0:2222", "ps1:2222"],
    "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
with tf.device(tf.replica_device_setter(cluster=cluster_spec)):
  # Build your graph
  v1 = tf.Variable(...)  # assigned to /job:ps/task:0
  v2 = tf.Variable(...)  # assigned to /job:ps/task:1
  v3 = tf.Variable(...)  # assigned to /job:ps/task:0
# Run compute
```
这个例子是没有指定参数`worker_device`和`ps_device`的,你可以手动指定
--------------------------------------------------------------------------------
继续代码注释,下面就是,模型的定义了
```python
      # Build model...variables and ops
      loss = ...
      global_step = tf.Variable(0)

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

      saver = tf.train.Saver()
      summary_op = tf.merge_all_summaries()
      init_op = tf.initialize_all_variables()

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/tmp/train_logs",
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target) as sess:
      # Loop until the supervisor shuts down or 1000000 steps have completed.
      step = 0
      while not sv.should_stop() and step < 1000000:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        _, step = sess.run([train_op, global_step])

    # Ask for all the services to stop.
    sv.stop()
```
考虑一个场景(`Between-graph`),我们有一个`parameter server`(存放着参数的副本),有好几个`worker server`(分别保存着相同的`graph`的副本).更通俗的说,我们有10台电脑,其中一台作为`parameter server`,其余九台作为`worker server`.因为同一个程序在10台电脑上同时运行(不同电脑,`job_name`,`task_index`不同),所以每个`worker server`上都有我们建立的`graph`的副本(`replica`).这时我们可以使用`Supervisor`帮助我们管理各个`process`.`Supervisor`的`is_chief`参数很重要,它指明用哪个`task`进行参数的初始化工作.`sv.managed_session(server.target)`创建一个被`sv`管理的`session`.
`is_chief`指定了哪个

```python
if __name__ == "__main__":
  tf.app.run()
```
To start the trainer with two parameter servers and two workers, use the following command line (assuming the script is called trainer.py):
```python
# On ps0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=0
# On ps1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=1
# On worker0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=0
# On worker1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=1
```
**可以看出,我们只需要写一个程序,在不同的主机上,传入不同的参数使其运行**


下篇将要介绍如何分布式计算深度学习模型
参考博客:
[1] http://weibo.com/ttarticle/p/show?id=2309403987407065210809
[2] http://weibo.com/ttarticle/p/show?id=2309403988813608274928
[3] http://blog.csdn.net/luodongri/article/details/52596780
[4]https://www.tensorflow.org/versions/r0.10/how_tos/distributed/index.html
[5]https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html#optimizers
[6]https://github.com/tensorflow/tensorflow/blob/5a5a25ea3ebef623e07fb9a46419a9df377a37a5/tensorflow/g3doc/api_docs/python/functions_and_classes/shard3/tf.train.SyncReplicasOptimizer.md
[7]https://github.com/tensorflow/tensorflow/blob/5a5a25ea3ebef623e07fb9a46419a9df377a37a5/tensorflow/g3doc/api_docs/python/functions_and_classes/shard6/tf.train.Supervisor.md#what-master-string-to-use
[8]https://www.tensorflow.org/versions/r0.11/api_docs/python/client.html#session-management
