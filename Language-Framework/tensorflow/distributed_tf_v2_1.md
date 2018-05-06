# 分布式Tensorflow(一)

## gRPC
分布式Tensorflow底层的通信是`gRPC`
gRPC首先是一个RPC,即`远程过程调用`,通俗的解释是:假设你在本机上执行一段代码`num=add(a,b)`,它调用了一个过程 `call`,然后返回了一个值`num`,你感觉这段代码只是在本机上执行的, 但实际情况是,本机上的`add`方法是将参数打包发送给服务器,然后服务器运行服务器端的`add`方法,返回的结果再将数据打包返回给客户端.

## tf.Session()
`Session`对象封装了执行`ops`和计算`Tensor` 的环境.`session`可能拥有计算机资源,为什么说可能呢:因为在分布式环境下`client`的`session`可能是不拥有计算机资源的.
```python
tf.Session.__init__(target='', graph=None, config=None):
```
**target**:
- '': 不使用`RPC`,请求了一个`In-process`的`session`
- 'local': 使用`PRC`,请求一个`In-process`的`sesson`,指定自己为`master`,`master`对分布式系统分配资源
- 'grpc://hostname:port':使用`RPC`,可以请求`In-process`或`not In-process`的`sessoin`

**config**:
- 接收`tf.ConfigProto`对象,一般是`tf.ConfigProto(allow_soft_placement=True,
        log_device_placement=True)`

## tf.train.Supervisor
`The Supervisor is a small wrapper around a Coordinator, a Saver, and a SessionManager that takes care of common needs of TensorFlow training programs.`

`Coordinator`: 管理多线程
`Saver` : 管理 `checkpoint`
`SessionManager`: 管理 `session`

在分布式环境下,`Supervisor`管理分布式系统中进程,如:
- 如果分布式系统中的一个进程出问题了,`sv.should_stop()`就会返回`True`
- 当`chief task`完成`各种初始化`(参数初始化...)后, `sv.managed_session()`才会返回`session`给其他`task`

**chief task:**
在分布式系统中, 必须有一个`task`被指定成`chief task`,`chief task`执行 `初始化`,`checkpoint`,`recovery`工作

## tf.train.syncReplicaOptimizer
**管理 `gradients`.**
这是 `Optimiser`的一个`Wraper`.通过这个`Wraper`实现分布式系统中同步问题.同步就是,所有`worker`执行一轮后,将所有的`gradient`保存起来,然后加起来(?为什么不平均...?),之后`apply` to `variable`.

**创建两个Queue**
- `N gradient queue`, `gradients`会被添加到`queues`中,然后`chief task`会执行`dequeue_many`,将`gradient`加起来,再`apply gradient to variable`
- 一个`token queue`,`apply gradient to variable`执行完后, `optimiser`将新的`global`值放入`queue`

**创建一个`variable`**  
- 每个`replica`创建一个 `local_step`,和`global step`比较,判断计算的`gradient`是否有效

**流程:**
This adds nodes to the graph to collect gradients and pause the trainers until variables are updated. For the PS:
1. A queue is created for each variable, and each replica now pushes the gradients into the queue instead of directly applying them to the variables.
2. For each gradient_queue, pop and sum the gradients once enough replicas (replicas_to_aggregate) have pushed gradients to the queue.
3. Apply the aggregated gradients to the variables.
4. Only after all variables have been updated, increment the global step.
5. Only after step 4, clear all the gradients in the queues as they are stale now (could happen when replicas are restarted and push to the queues multiple times, or from the backup replicas).
6. Only after step 5, pushes global_step in the token_queue, once for each worker replica. The workers can now fetch it to its local_step variable and start the next batch.

For the replicas:
1. Start a step: fetch variables and compute gradients.
2. Once the gradients have been computed, push them into gradient_queue only if local_step equals global_step, otherwise the gradients are just dropped. This avoids stale gradients.
3. After pushing all the gradients, dequeue an updated value of global_step from the token queue and record that step to its local_step variable. Note that this is effectively a barrier.
4. Start the next batch.

**两个队列的初始化,需要手动完成**

**Usage**
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

#In the training program, every worker will run the train_op as if not synchronized. But
#one worker (usually the chief) will need to execute the chief_queue_runner and
#get_init_tokens_op generated from this optimizer.
# After the session is created by the Supervisor and before the main while
# loop:
if is_chief and FLAGS.sync_replicas:
  sv.start_queue_runners(sess, [chief_queue_runner])
  # Insert initial tokens to the queue.
  sess.run(init_token_op)
```
## global_step
在分布式代码编写的时候,我们需要定义一个`global_step`变量(记得将trainable=False),这个变量是存在`paramter server`上的,`paramter server`上的参数每更新一次,`gloable_step`就会`+1`.在定义`supevisor`时候,将`globale_step`传入,这样你定义的`global_step`就会有这个功能
