## `In-graph` 与 `Between-graph`
`In-graph`:只有一个`graph`的`replica`,通过`tf.Session("grpc_address")`将一个`worker`指定为`master`,`master`进行`cluster`中的资源分配

`Betweent-graph`:每个`设备`都有一个`graph`的`replica`,各个`设备`之间并行计算,互不打扰.`variable`存放在`parameter server`上

## `Synchronous` 与 `Asynchronous`
在 `Between-graph`条件下:
`Asynchronous`: 每个设备独立计算`gradient`,然后独立地`apply gradients to variables`.假设有`N-replicas`,全局进行一次运算,`variable`会被更新 `N次`

`Synchronous`: 每个设备独立计算`gradient`,然后汇集到一起处理,处理之后,将处理后的值一次性交给`variable`.

在`In-graph`条件下:
由于这种方法不常用,在此就不介绍了,下面也都是介绍`Between-graph`


```{mermaid}
graph TB
  subgraph ps
    parameter_server
  end
  subgraph worker:
  gradients_queue
  worker0
  worker1
  worker2
  ...
  worker0 --> |gradient|gradients_queue
  worker1 --> |gradient|gradients_queue
  worker2 --> |gradient|gradients_queue
  ... --> |gradient|gradients_queue
  end

  parameter_server --> |variables|worker0
  parameter_server --> |variables|worker1
  parameter_server --> |variables|worker2
  parameter_server --> |variables|...

  gradients_queue --> |gradient|parameter_server
```
<center>Between-graph,同步</center>
```{mermaid}
graph LR

  subgraph worker:
    worker0
    worker1
    worker2
    ...
  end
  subgraph ps
    parameter_server
  end
  parameter_server --> |variables|worker0
  parameter_server --> |variables|worker1
  parameter_server --> |variables|worker2
  parameter_server --> |variables|...
  worker0 --> |gradient|parameter_server
  worker1 --> |gradient|parameter_server
  worker2 --> |gradient|parameter_server
  ... --> |gradient|parameter_server
```
<center>Between-graph,异步</center>
## Supervisor 在分布式中的应用
**管理`session`, 协调多进程, 管理`saver`**
**管理多线程,如果有一个线程出错,sv.should_stop()返回True**
A training helper that checkpoints models and computes summaries.

The Supervisor is a small wrapper around a Coordinator, a Saver, and a SessionManager that takes care of common needs of TensorFlow training programs.


Use for a single program
```python
with tf.Graph().as_default():
  ...add operations to the graph...
  # Create a Supervisor that will checkpoint the model in '/tmp/mydir'.
  sv = Supervisor(logdir='/tmp/mydir')
  # Get a TensorFlow session managed by the supervisor.
  with sv.managed_session(FLAGS.master) as sess:
    # Use the session to train the graph.
    while not sv.should_stop():
      sess.run(<my_train_op>)
```
Within the with sv.managed_session() block all variables in the graph have been initialized. In addition, a few services have been started to checkpoint the model and add summaries to the event log.

If the program crashes and is restarted, the managed session automatically reinitialize variables from the most recent checkpoint.

The supervisor is notified of any exception raised by one of the services. After an exception is raised, should_stop() returns True. In that case the training loop should also stop. This is why the training loop has to check for sv.should_stop().

Exceptions that indicate that the training inputs have been exhausted, tf.errors.OutOfRangeError, also cause sv.should_stop() to return True but are not re-raised from the with block: they indicate a normal termination.
To train with replicas you deploy the same program in a `Cluster`. One of the tasks must be identified as the **`chief`**: the task that handles `initialization`, `checkpoints`, `summaries`, and `recovery`. The other tasks depend on the chief for these services.
The only change you have to do to the single program code is to indicate if the program is running as the chief.
```python
# Choose a task as the chief. This could be based on server_def.task_index,
# or job_def.name, or job_def.tasks. It's entirely up to the end user.
# But there can be only one *chief*.
is_chief = (server_def.task_index == 0)
server = tf.train.Server(server_def)

with tf.Graph().as_default():
  ...add operations to the graph...
  # Create a Supervisor that uses log directory on a shared file system.
  # Indicate if you are the 'chief'
  sv = Supervisor(logdir='/shared_directory/...', is_chief=is_chief)
  # Get a Session in a TensorFlow server on the cluster.
  with sv.managed_session(server.target) as sess:
    # Use the session to train the graph.
    while not sv.should_stop():
      sess.run(<my_train_op>)
```
In the `chief` task, the Supervisor works exactly as in the first example above. In the other tasks `sv.managed_session()` waits for the Model to have been initialized before returning a session to the training code. The non-chief tasks depend on the chief task for initializing the model.

If one of the tasks crashes and restarts, managed_session() checks if the Model is initialized. If yes, it just creates a session and returns it to the training code that proceeds normally. If the model needs to be initialized, the chief task takes care of reinitializing it; the other tasks just wait for the model to have been initialized.

NOTE: This modified program still works fine as a single program. The single program marks itself as the chief.

What `master` string to use

Whether you are running on your machine or in the cluster you can use the following values for the --master flag:

Specifying '' requests an in-process session that does not use RPC.

Specifying 'local' requests a session that uses the RPC-based "Master interface" to run TensorFlow programs. See tf.train.Server.create_local_server() for details.

Specifying 'grpc://hostname:port' requests a session that uses the RPC interface to a specific , and also allows the in-process master to access remote tensorflow workers. Often, it is appropriate to pass server.target (for some tf.train.Server named `server`).

Advanced use

Launching additional services

managed_session() launches the Checkpoint and Summary services (threads). If you need more services to run you can simply launch them in the block controlled by managed_session().

Example: Start a thread to print losses. We want this thread to run every 60 seconds, so we launch it with sv.loop().
```python
  ...
  sv = Supervisor(logdir='/tmp/mydir')
  with sv.managed_session(FLAGS.master) as sess:
    sv.loop(60, print_loss, (sess))
    while not sv.should_stop():
      sess.run(my_train_op)
```
Launching fewer services

managed_session() launches the "summary" and "checkpoint" threads which use either the optionally summary_op and saver passed to the constructor, or default ones created automatically by the supervisor. If you want to run your own summary and checkpointing logic, disable these services by passing None to the summary_op and saver parameters.

Example: Create summaries manually every 100 steps in the chief.
```python
  # Create a Supervisor with no automatic summaries.
  sv = Supervisor(logdir='/tmp/mydir', is_chief=is_chief, summary_op=None)
  # As summary_op was None, managed_session() does not start the
  # summary thread.
  with sv.managed_session(FLAGS.master) as sess:
    for step in xrange(1000000):
      if sv.should_stop():
        break
      if is_chief and step % 100 == 0:
        # Create the summary every 100 chief steps.
        sv.summary_computed(sess, sess.run(my_summary_op))
      else:
        # Train normally
        sess.run(my_train_op)
```
Custom model initialization

managed_session() only supports initializing the model by running an init_op or restoring from the latest checkpoint. If you have special initialization needs, see how to specify a local_init_op when creating the supervisor. You can also use the SessionManager directly to create a session and check if it could be initialized automatically.

## tf.train.SyncReplicasOptimizer 进行同步训练
`tf.train.Supervisor`处理`cluster`中参数的初始化,进程之间的问题.`tf.train.SyncReplicasOptimizer`处理`gradients的同步问题`
Class to synchronize, aggregate gradients and pass them to the optimizer.

In a typical asynchronous training environment, it's common to have some stale gradients. For example, with a N-replica asynchronous training, gradients will be applied to the variables N times independently. Depending on each replica's training speed, some gradients might be calculated from copies of the variable from several steps back (N-1 steps on average). This optimizer avoids stale gradients by collecting gradients from all replicas, summing them, then applying them to the variables in one shot, after which replicas can fetch the new variables and continue.

The following queues are created:

N gradient queues, one per variable to train. Gradients are pushed to these queues and the chief worker will dequeue_many and then sum them before applying to variables.
1 token queue where the optimizer pushes the new global_step value after all gradients have been applied.
The following variables are created:

N local_step, one per replica. Compared against global step to check for staleness of the gradients.
This adds nodes to the graph to collect gradients and pause the trainers until variables are updated. For the `PS`: 1. A queue is created for each variable, and each replica now pushes the gradients into the queue instead of directly applying them to the variables. 2. For each gradient_queue, pop and sum the gradients once enough replicas (replicas_to_aggregate) have pushed gradients to the queue. 3. Apply the aggregated gradients to the variables. 4. Only after all variables have been updated, increment the global step. 5. Only after step 4, clear all the gradients in the queues as they are stale now (could happen when replicas are restarted and push to the queues multiple times, or from the backup replicas). 6. Only after step 5, pushes global_step in the token_queue, once for each worker replica. The workers can now fetch it to its local_step variable and start the next batch.

For the `replicas`: 1. Start a step: fetch variables and compute gradients. 2. Once the gradients have been computed, push them into gradient_queue only if local_step equals global_step, otherwise the gradients are just dropped. This avoids stale gradients. 3. After pushing all the gradients, dequeue an updated value of global_step from the token queue and record that step to its local_step variable. Note that this is effectively a barrier. 4. Start the next batch.

Usage
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
In the training program, every worker will run the train_op as if not synchronized. But one worker (usually the chief) will need to execute the chief_queue_runner and get_init_tokens_op generated from this optimizer.

# After the session is created by the Supervisor and before the main while
# loop:
if is_chief and FLAGS.sync_replicas:
  sv.start_queue_runners(sess, [chief_queue_runner])
  # Insert initial tokens to the queue.
  sess.run(init_token_op)
```
