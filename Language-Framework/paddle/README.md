# 框架总览

* `Variable`
* `Parameter` : `persistable=True` 的 `Variable`, 不同的 `iteration` 之间, 状态会被保留, `Parameter` 是在 `global block` 下创建的



* `fluid.Program()` : 执行的最小单位, 可以看做为子图
  * `fluid.default_startup_program()` : 模型变量的初始化 由此 `program` 负责
  * `fluid.default_main_program()`: 其它 op 由此 `program` 负责
  * 创建的`Parameter`都在当前 `program` 的 `global_block()` 下
* `Block` : c++ 中作用域的概念, 一个 `Program` 由多个 `Block` 构成
  * `if else Block`, `switch case Block`, `while Block`
* `fluid.Executor(place=fluid.CPUPlace())`
  * `Executor` 核心执行模块, 负责编译 `program` 并执行, 一次执行整个 `program` , 这个和 `tensorflow` 有区别, `tensorflow` 每次只是执行和 `fetch` 相关的子图

```python
# 手动搞 program, 当然使用默认的也是可以的
import paddle.fluid as fluid
import numpy

train_program = fluid.Program()
startup_program = fluid.Program()
with fluid.program_guard(train_program, startup_program):
    data = fluid.layers.data(name='X', shape=[1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = fluid.layers.mean(hidden)
    sgd = fluid.optimizer.SGD(learning_rate=0.001)
    sgd.minimize(loss)

use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

# Run the startup program once and only once.
# Not need to optimize/compile the startup program.
startup_program.random_seed=1
exe.run(startup_program)

# Run the main program directly without compile.
x = numpy.random.random(size=(10, 1)).astype('float32')
loss_data, = exe.run(train_program,
                     feed={"X": x},
                     fetch_list=[loss.name])

# Or use CompiledProgram:
compiled_prog = compiler.CompiledProgram(train_program)
loss_data, = exe.run(compiled_prog,
             feed={"X": x},
             fetch_list=[loss.name])
```



* `fluid.ParamAttr(name=None, initializer=None, learning_rate=1.0, regularizer=None, trainable=True, gradient_clip=None, do_model_average=False)` [link](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/ParamAttr_cn.html#paramattr)
  * 参数共享由 `name` 相同所实现. 

* `fluid.Scope()` : 存放 `Variable` 的地方, 包含了name与Variable的映射
  * 为啥在 `inference` 的时候需要搞个 `Scope` 呢?

```python
import numpy

new_scope = fluid.Scope()
# 这是 global_scope 返回的就是 new_scope 了
with fluid.scope_guard(new_scope):
     fluid.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), fluid.CPUPlace())
numpy.array(new_scope.find_var("data").get_tensor())
```



# 输入流水线

* `reader`
  * `paddle` 中 `reader` 的定义仅仅是是一个 `Python iterator`, 一次返回一个 样本的 `iterator`

```python
def reader_creator_random_image_and_label(width, height, label):
    def reader():
        while True:
            yield numpy.random.uniform(-1, 1, size=width*height), label
    return reader

reader = reader_creator_random_image_and_label(32, 32, 1) #迭代器, 一调用就yield
```

* `paddle.batch()`
  * 迭代器作为参数, 构建 `batch`

```python
mnist_train = paddle.dataset.mnist.train()
mnist_train_batch_reader = paddle.batch(mnist_train, 128)
```

* 自定义 `batch_reader` 
  * 这玩意和自定义 `reader` 是一个逻辑的

```python
def custom_batch_reader():
    while True:
        batch = []
        for i in xrange(128):
            batch.append((numpy.random.uniform(-1, 1, 28*28),)) # note that it's a tuple being appended.
        yield batch

mnist_random_image_batch_reader = custom_batch_reader
```



* 如何将 `reader` 和 模型的输入对应起来

```python
image_layer = paddle.layer.data("image", ...)
label_layer = paddle.layer.data("label", ...)

# 通过 train 将 batch reader 与 data_layer 对应起来
batch_reader = paddle.batch(paddle.dataset.mnist.train(), 128)
paddle.train(batch_reader, {"image":0, "label":1}, 128, 10, ...)
```



### data reader 装饰器

> 装饰 data reader, 给予其更强大的功能

```python
# 预取数据装饰器
buffered_reader = paddle.reader.buffered(paddle.dataset.mnist.train(), 100)

# random_shuffle, 缓存512个, 然后 shuffle, 这个和 预取的顺序应该是怎么样的
reader = paddle.reader.shuffle(paddle.dataset.mnist.train(), 512)
```



### DataLoader

> 异步数据读取

* 三类数据源
  * `set_sample_generator()`, 要求数据源返回的数据格式为 `[image1, label1]`
  * `set_sample_list_generator()` , 要求数据源返回的数据格式为 `[(img1, lable1), (img2, label2), ....]` , 单样本构成的 `list`
    * 这玩意怎么用啊, 一次给模型几个???
  * `set_batch_generator()`, 要求数据源返回的数据格式为 `[batch_imgs, batch_labels]` , 返回的是个 `batch` 数据

```python
import paddle
import paddle.fluid as fluid

ITERABLE = True
USE_CUDA = True
USE_DATA_PARALLEL = True

if ITERABLE:
    # 若DataLoader可迭代，则必须设置places参数
    if USE_DATA_PARALLEL:
        # 若进行多GPU卡训练，则取所有的CUDAPlace
        # 若进行多CPU核训练，则取多个CPUPlace，本例中取了8个CPUPlace
        places = fluid.cuda_places() if USE_CUDA else fluid.cpu_places(8)
    else:
        # 若进行单GPU卡训练，则取单个CUDAPlace，本例中0代表0号GPU卡
        # 若进行单CPU核训练，则取单个CPUPlace，本例中1代表1个CPUPlace
        places = fluid.cuda_places(0) if USE_CUDA else fluid.cpu_places(1)
else:
    # 若DataLoader不可迭代，则不需要设置places参数
    places = None

# 使用sample级的reader作为DataLoader的数据源
data_loader1 = fluid.io.DataLoader.from_generator(feed_list=[image1, label1], capacity=10, iterable=ITERABLE)
data_loader1.set_sample_generator(fake_sample_reader, batch_size=32, places=places)

# 使用sample级的reader + fluid.io.batch设置DataLoader的数据源
data_loader2 = fluid.io.DataLoader.from_generator(feed_list=[image2, label2], capacity=10, iterable=ITERABLE)
sample_list_reader = fluid.io.batch(fake_sample_reader, batch_size=32)
sample_list_reader = fluid.io.shuffle(sample_list_reader, buf_size=64) # 还可以进行适当的shuffle
data_loader2.set_sample_list_generator(sample_list_reader, places=places)

# 使用batch级的reader作为DataLoader的数据源
data_loader3 = fluid.io.DataLoader.from_generator(feed_list=[image3, label3], capacity=10, iterable=ITERABLE)
data_loader3.set_batch_generator(fake_batch_reader, places=places)
```

