
## `distributed tensorflow`的一些基本概念
一个 TensorFlow 图描述了计算的过程. 为了进行计算, 图必须在`session`(会话)里被启动. `session`将图的 `op`分发到诸如`CPU`或`GPU`之类的`设备`上
TF采用了PS/Worker的结构来定义集群，其中
PS(parameter server)：存储variable（模型参数），主要负责参数更新和发放；
Worker：存储operator，主要负责图计算和梯度计算（TF使用Optimizer实现了自动化的梯度计算）；
job：由于工作类型不同，用job_name来区分ps和worker
task：对于每个worker来说，具体做什么任务（算什么图）也有可能不同，用task_index区分
device：指具体的CPU/GPU，通常PS绑定到CPU上，Worker绑定到GPU上，各取所长。
syncReplicaOptimizer：同步优化器，其本质仍然是用普通优化器进行梯度计算，但是通过Queue机制和Coordinator多线程协同实现了所有worker的梯度汇总和平均，最终将梯度传回PS进行参数更新。
client:相对于服务器而言.
## Launching the graph in a distributed session
To create a TensorFlow cluster, launch a TensorFlow server on each of the machines in the cluster. When you instantiate a Session in your client, you pass it the network location of one of the machines in the cluster:
```python
with tf.Session("grpc://example.org:2222") as sess:
  # Calls to sess.run(...) will be executed on the cluster.
  ...
```
This machine becomes the master for the session. The master distributes the graph across other machines in the cluster (workers), much as the local implementation distributes the graph across available compute resources within a machine.
**服务器怎么实现呢?**

You can use `with tf.device():` statements to directly specify workers for particular parts of the graph:
```python
with tf.device("/job:ps/task:0"):
  weights = tf.Variable(...)
  biases = tf.Variable(...)
```
```python
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#这样会打印各个op 和 variable的分配情况
```
## Q1:如何写`In-graph`程序?
我尝试过一种思想, 写`a`个`server`,然后通过`server.join()`暴露`gprc`接口,然后客户端使用`tf.device()`指定对应的设备进行运算.行不通!
错误例子:
```python
#server端代码
import tensorflow as tf

job_tasks = {"ps":[...,...],"worker":[...,...]}
cluster = tf.train.ClusterSpec(job_tasks)
server = tf.trian.Server(cluster=cluster, job_name="current_job_name",
                          task_index="current_task_index")
server.join()
```
```python
#client 代码
job_tasks = {"ps":[...,...],"worker":[...,...]}
cluster = tf.train.ClusterSpec(job_tasks)

with tf.device("/job:ps/task:0"):
  w = tf.get_variable("w",shape=[2,4],dtype=tf.flaot32)
  b = tf.get_variable("b",shape=[4],dtype=tf.flaot32)
with tf.device("/job:worker/task:0"):
  x = tf.placeholder(dtype=tf.float32, shape=[1,2])
  res = tf.matmul(x, w)
  init = tf.initialize_all_variables()
with tf.Session() as sess:
  sess.run(init)
  print sess.run(res)
```
运行的时候,这段代码会报错.如果将`tf.Session()`改写成`tf.Session(worker0.example.com:port)`就可以运行的很好.对于`tf.Session(worker0.example.com:port)`,我的理解是:由于`tensorflow`分布式底层是通信是`grpc`,`client`执行这段代码的时候,是将数据打包送给`worker0.example.com:port`执行的,然后服务器端运行完,又把数据返回`client`.

```{}
```
## error that i have encounted
**Error1: Distributed Tensorflow, Value Error "When using replicas, all Variables must have their device set" name:"Variable"**
Ans1: `tf.get_variable`的定义放在了`with tf.device()`外,应该写在里面
Ans2: `tf.train.Supervisor`要在`with tf.device()`外定义
