# tensorflow 如何读取数据
tensorflow有三种把数据放入计算图中的方式:
* 通过feed_dict
* 通过文件名读取数据:一个输入流水线 在计算图的开始部分从文件中读取数据
* 把数据预加载到一个常量或者变量中

第一个和第三个都很简单,本文主要介绍的是第二种.
考虑一个场景:我们有大量的数据,无法一次导入内存,那我们一次就只能导入几个`nimi-batch`,然后进行训练,然后再导入几个`mini-batch`然后再进行训练.可能你会想,为什么我们不能在训练的时候,并行的导入下次要训练的几个`mini-batch`呢?幸运的是,tensorflow已经提供了这个机制.也许你还会问,既然你可以在训练前个`mini-batch`的时候把要训练的下几个`mini-batch`导进来,那么内存是足够将两次的`mini-batch`都导入进来的,为什么我们不直接把两次的`mini-batch`都导入呢,占满整个内存.实际上,这种方法,相比之前所述的流水线似的方法,还是慢的.

现在来看tensorflow给我们提供了什么

## Queue

`Queue`,队列,用来存放数据(跟`Variable`似的),`tensorflow`中的`Queue`中已经实现了同步机制,所以我们可以放心的往里面添加数据还有读取数据.如果`Queue`中的数据满了,那么`en_queue`操作将会阻塞,如果`Queue`是空的,那么`dequeue`操作就会阻塞.在常用环境中,一般是有多个`en_queue`线程同时像`Queue`中放数据,有一个`dequeue`操作从`Queue`中取数据.一般来说`enqueue`线程就是准备数据的线程,`dequeue`线程就是训练数据的线程.
![](https://www.tensorflow.org/images/IncremeterFifoQueue.gif)


## Coordinator(协调者)
`Coordinator`就是用来帮助多个线程同时停止.线程组需要一个`Coordinator`来协调它们之间的工作.
```python
# Thread body: loop until the coordinator indicates a stop was requested.
# If some condition becomes true, ask the coordinator to stop.
#将coord传入到线程中,来帮助它们同时停止工作
def MyLoop(coord):
  while not coord.should_stop():
    ...do something...
    if ...some condition...:
      coord.request_stop()

# Main thread: create a coordinator.
coord = tf.train.Coordinator()

# Create 10 threads that run 'MyLoop()'
threads = [threading.Thread(target=MyLoop, args=(coord,)) for i in xrange(10)]

# Start the threads and wait for all of them to stop.
for t in threads:
  t.start()
coord.join(threads)
```

## QueueRunner
`QueueRunner`创建多个线程对`Queue`进行`enqueue`操作.它是一个`op`.这些线程可以通过上面所述的`Coordinator`来协调它们同时停止工作.
```python
example = ...ops to create one example...
# Create a queue, and an op that enqueues examples one at a time in the queue.
queue = tf.RandomShuffleQueue(...)
enqueue_op = queue.enqueue(example)
# Create a training graph that starts by dequeuing a batch of examples.
inputs = queue.dequeue_many(batch_size)
train_op = ...use 'inputs' to build the training part of the graph...
```
```python
# Create a queue runner that will run 4 threads in parallel to enqueue
# examples.
qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)

# Launch the graph.
sess = tf.Session()
# Create a coordinator, launch the queue runner threads.
coord = tf.train.Coordinator()
#执行 enqueue线程,像queue中放数据
enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
# Run the training loop, controlling termination with the coordinator.
for step in xrange(1000000):
    if coord.should_stop():
        break
    sess.run(train_op)
# When done, ask the threads to stop.
coord.request_stop()
# And wait for them to actually do it.
coord.join(enqueue_threads)
```

**有了这些基础,我们来看一下tensorflow的`input-pipeline`**
## tensorflow 输入流水线
我们先梳理一些之前说的东西.`Queue`是一个队列,`QueueRunner`用来创建多个线程对`Queue`进行`enqueue`操作.`Coordinator`可用来协调`QueueRunner`创建出来的线程共同停止工作.

下面来看tensorflow的输入流水线.
1. 准备文件名
2. 定义文件中数据的解码规则
3. 读取文件中数据

从文件里读数据,读完了,就换另一个文件.文件名放在`string_input_producer`中.
```python
import tensorflow as tf
#一个Queue,用来保存文件名字.对此Queue,只读取,不dequeue
filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])

#用来从文件中读取数据, LineReader,每次读一行
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4])

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  #在调用run或eval执行读取之前，必须
  #用tf.train.start_queue_runners来填充队列
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(10):
    # Retrieve a single instance:
    example, label = sess.run([features, col5])
    print(example, label)
  coord.request_stop()
  coord.join(threads)
```
