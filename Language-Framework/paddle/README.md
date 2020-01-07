# 框架总览

* `Variable`
* `Parameter` : `persistable=True` 的 `Variable`, 不同的 `iteration` 之间, 状态会被保留, `Parameter` 是在 `global block` 下创建的

> 感觉`Parameter(Variable)` 是和 `block` 独立的东西. 甚至是和 `program` 独立的东西. `Program` 存的只是`op` 而已, 存的是 `Variable` 读,写 `op`. 真正的变量是在 `Scope` 中的.



* `fluid.Program()` : 执行的最小单位, 可以看做为子图
  * `fluid.default_startup_program()` : 模型变量的初始化 由此 `program` 负责
  * `fluid.default_main_program()`: 其它 op 由此 `program` 负责
  * 创建的`Parameter`都在当前 `program` 的 `global_block()` 下
* `Block` : c++ 中作用域的概念, 一个 `Program` 由多个 `Block` 构成
  * `if else Block`, `switch case Block`, `while Block`
  * 所以整体结构为: 
    * 整体的计算图由多个Program构成
    * Program中有很多Block
    * Block中存在很多operation
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

* `fluid.Scope()` : 应该是存了所有的 `persistable` 的变量, 可以确定的是

```python
import numpy

new_scope = fluid.Scope()
# 这是 global_scope 返回的就是 new_scope 了
with fluid.scope_guard(new_scope):
     fluid.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), fluid.CPUPlace())
numpy.array(new_scope.find_var("data").get_tensor())
```

```python
import paddle.fluid as fluid
import numpy as np
main_prog = fluid.Program()
startup_prog = fluid.Program()
with fluid.program_guard(main_prog, startup_prog):
    data = fluid.layers.data(name="img", shape=[64, 784], append_batch_size=False)
    w = fluid.layers.create_parameter(shape=[784, 200], dtype='float32', name='fc_w')
    b = fluid.layers.create_parameter(shape=[200], dtype='float32', name='fc_b')
    hidden_w = fluid.layers.matmul(x=data, y=w)
    hidden_b = fluid.layers.elementwise_add(hidden_w, b)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_prog)

for block in main_prog.blocks:
    for param in block.all_parameters():
        pd_var = fluid.global_scope().find_var(param.name)
        pd_param = pd_var.get_tensor()
        print("load: {}, shape: {}".format(param.name, param.shape))
        print("Before setting the numpy array value: {}".format(np.array(pd_param).ravel()[:5]))
        pd_param.set(np.ones(param.shape), place)
        print("After setting the numpy array value: {}".format(np.array(pd_param).ravel()[:5]))
```







# 解析 fluid.layers.fc

```python
def fc(input,
       size,
       num_flatten_dims=1,
       param_attr=None,
       bias_attr=None,
       act=None,
       is_test=False,
       name=None):

    helper = LayerHelper("fc", **locals())

    dtype = helper.input_dtype()

    mul_results = []
    for input_var, param_attr in helper.iter_inputs_and_params():
        input_shape = input_var.shape
        param_shape = [
            reduce(lambda a, b: a * b, input_shape[num_flatten_dims:], 1)
        ] + [size]

        w = helper.create_parameter(
            attr=param_attr, shape=param_shape, dtype=dtype, is_bias=False)
        tmp = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="mul",
            inputs={"X": input_var,
                    "Y": w},
            outputs={"Out": tmp},
            attrs={"x_num_col_dims": num_flatten_dims,
                   "y_num_col_dims": 1})
        mul_results.append(tmp)

    if len(mul_results) == 1:
        pre_bias = mul_results[0]
    else:
        pre_bias = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="sum",
            inputs={"X": mul_results},
            outputs={"Out": pre_bias},
            attrs={"use_mkldnn": False})
    # add bias
    pre_activation = helper.append_bias_op(pre_bias, dim_start=num_flatten_dims)
    # add activation
    return helper.append_activation(pre_activation)


```



**LayerHelper**

```python
class LayerHelper(LayerHelperBase):
    def __init__(self, layer_type, **kwargs):
        self.kwargs = kwargs
        name = self.kwargs.get('name', None)
        if name is None:
            self.kwargs['name'] = unique_name.generate(layer_type)

        super(LayerHelper, self).__init__(
            self.kwargs['name'], layer_type=layer_type)
		# 这里可以看到, op 是添加到 main_program.current_block() 上的
    def append_op(self, *args, **kwargs):
        return self.main_program.current_block().append_op(*args, **kwargs)

    def multiple_input(self, input_param_name='input'):
        inputs = self.kwargs.get(input_param_name, [])
        ret = []
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            for inp in inputs:
                ret.append(self.to_variable(inp))
        else:
            ret.append(self.to_variable(inputs))
        return ret

    def input(self, input_param_name='input'):
        inputs = self.multiple_input(input_param_name)
        if len(inputs) != 1:
            raise "{0} layer only takes one input".format(self.layer_type)
        return inputs[0]

    @property
    def param_attr(self):
        return ParamAttr._to_attr(self.kwargs.get('param_attr', None))

    @property
    def bias_attr(self):
        return ParamAttr._to_attr(self.kwargs.get('bias_attr', None))

    #TODO (jiabin): reconstruct this in LayerObjHelper and avoid dependency of param_attr
    def multiple_param_attr(self, length):
        param_attr = self.param_attr
        if isinstance(param_attr, ParamAttr):
            param_attr = [param_attr]

        if len(param_attr) != 1 and len(param_attr) != length:
            raise ValueError("parameter number mismatch")
        elif len(param_attr) == 1 and length != 1:
            tmp = [None] * length
            for i in six.moves.range(length):
                tmp[i] = copy.deepcopy(param_attr[0])
            param_attr = tmp
        return param_attr

    def iter_inputs_and_params(self, input_param_name='input'):
        inputs = self.multiple_input(input_param_name)
        param_attrs = self.multiple_param_attr(len(inputs))
        for ipt, param_attr in zip(inputs, param_attrs):
            yield ipt, param_attr

    def input_dtype(self, input_param_name='input'):
        inputs = self.multiple_input(input_param_name)
        dtype = None
        for each in inputs:
            if dtype is None:
                dtype = each.dtype
            elif dtype != each.dtype:
                raise ValueError("Data Type mismatch: %d to %d" %
                                 (dtype, each.dtype))
        return dtype
		# parameter 是放在 main_program 的 global_block 中的
    def get_parameter(self, name):
        param = self.main_program.global_block().var(name)
        if not isinstance(param, Parameter):
            raise ValueError("no Parameter name %s found" % name)
        return param

    #TODO (jiabin): reconstruct this in LayerObjHelper and avoid dependency of bias_attr
    def append_bias_op(self, input_var, dim_start=1, dim_end=None):
        """
        Append bias operator and return its output. If the user does not set
        bias_attr, append_bias_op will return input_var

        :param input_var: the input variable. The len(input_var.shape) is
        larger or equal than 2.
        :bias_initializer: an instance of a subclass of Initializer used to
        initialize the bias
        :param dim_start:
        :param dim_end: the shape of the bias will be
        input_var.shape[dim_start:dim_end]. The bias is broadcasted to other
        dimensions and added to input_var to get the output
        """
        size = list(input_var.shape[dim_start:dim_end])
        bias_attr = self.bias_attr
        if not bias_attr:
            return input_var

        b = self.create_parameter(
            attr=bias_attr, shape=size, dtype=input_var.dtype, is_bias=True)
        tmp = self.create_variable_for_type_inference(dtype=input_var.dtype)
        self.append_op(
            type='elementwise_add',
            inputs={'X': [input_var],
                    'Y': [b]},
            outputs={'Out': [tmp]},
            attrs={'axis': dim_start})
        return tmp

    #TODO (jiabin): reconstruct this in LayerObjHelper and avoid dependency of act
    def append_activation(self, input_var):
        act = self.kwargs.get('act', None)
        if act is None:
            return input_var
        if isinstance(act, six.string_types):
            act = {'type': act}
        else:
            raise TypeError(str(act) + " should be unicode or str")

        if 'use_cudnn' in self.kwargs and self.kwargs.get('use_cudnn'):
            act['use_cudnn'] = self.kwargs.get('use_cudnn')
        if 'use_mkldnn' in self.kwargs:
            act['use_mkldnn'] = self.kwargs.get('use_mkldnn')
        act_type = act.pop('type')

        tmp = self.create_variable_for_type_inference(dtype=input_var.dtype)
        self.append_op(
            type=act_type,
            inputs={"X": [input_var]},
            outputs={"Out": [tmp]},
            attrs=act)
        return tmp

    #TODO (jiabin): should we remove this since it has never be used
    def _get_default_initializer(self, dtype):
        if dtype is None or dtype_is_floating(dtype) is True:
            return Xavier()
        else:
            # For integer and boolean types, initialize with all zeros
            return Constant()

    #TODO (jiabin): reconstruct this in LayerObjHelper and avoid dependency of kwargs
    def is_instance(self, param_name, cls):
        param = self.kwargs.get(param_name, None)
        if not isinstance(param, cls):
            raise TypeError("The input {0} parameter of method {1} must be {2}",
                            param_name, self.layer_type, cls.__name__)

```



# 几个上下文管理器

```python
# 重置唯一名字计数器, 在相同的 program 中, 会起到参数复用的效果?
with fluid.unique_name.guard():
    with fluid.program_gurad(test_program, fluid.Program()):
```



# 输入流水线

[数据预处理](https://www.paddlepaddle.org.cn/documentation/docs/zh/user_guides/howto/prepare_data/reader_cn.html#batch-reader-readerbatch-size)

* `reader` (函数迭代器)
  * `paddle` 中 `reader` 的定义仅仅是是一个 `Python iterator`, 一次返回一个 样本的 `iterator`

```python
def reader_creator_random_image_and_label(width, height, label):
    def reader():
        while True:
            yield numpy.random.uniform(-1, 1, size=width*height), label
    return reader

reader = reader_creator_random_image_and_label(32, 32, 1) #迭代器, 一调用就yield
```

* `paddle.batch()`    (纯python实现, 也是函数迭代器)
  * 迭代器作为参数, 构建 `batch`

```python
mnist_train = paddle.dataset.mnist.train()
mnist_train_batch_reader = paddle.batch(mnist_train, 128)
```

* 自定义 `batch_reader`  (也是个函数迭代器) : **一次返回一个`batch` 的 `reader` **
  * 这玩意和自定义 `reader` 是一个逻辑的
  * [batch_reader VS reader ](https://www.paddlepaddle.org.cn/documentation/docs/zh/user_guides/howto/prepare_data/reader_cn.html#readermini-batch)

```python
def custom_batch_reader():
    while True:
        batch = []
        for i in xrange(128):
            batch.append((numpy.random.uniform(-1, 1, 28*28),)) # note that it's a tuple being appended.
        yield batch

mnist_random_image_batch_reader = custom_batch_reader
```



* 如何将 `reader` 和 模型的输入对应起来, **这种方式在实际使用中用的很少**

```python
image_layer = paddle.layer.data("image", ...)
label_layer = paddle.layer.data("label", ...)

# 通过 train 将 batch reader 与 data_layer 对应起来
batch_reader = paddle.batch(paddle.dataset.mnist.train(), 128)
paddle.train(batch_reader, {"image":0, "label":1}, 128, 10, ...)
```



### data reader 装饰器

> 装饰 data reader, 给予其更强大的功能, 都是纯python实现

```python
# 预取数据装饰器
buffered_reader = paddle.reader.buffered(paddle.dataset.mnist.train(), 100)

# random_shuffle, 缓存512个, 然后 shuffle, 这个和 预取的顺序应该是怎么样的
reader = paddle.reader.shuffle(paddle.dataset.mnist.train(), 512)
```



### DataLoader

> 异步数据读取, reader加上装饰器之后shuffle东西的的话应该也是异步的吧.

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

# 使用sample级的reader作为DataLoader的数据源, 这里是data_loader 与 计算图的绑定
data_loader1 = fluid.io.DataLoader.from_generator(feed_list=[image1, label1], capacity=10, iterable=ITERABLE)
# 这里是数据与data_loader的绑定
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



**其它输入流水线工具**

```python
image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
reader = fluid.layers.create_py_reader_by_data(capacity=64,
                                               feed_list=[image, label])
reader.decorate_paddle_reader(
    paddle.reader.shuffle(paddle.batch(mnist.train(), batch_size=5), buf_size=500))
```



## paddle输入流水线的N种操作

1. `placeholder + run 时候的 feedlist`
2. `reader` 与 `paddle.train`
3. `py_reader`
   * 如果输入比较复杂, 包含`lod-tensor`和非 `lod-tensor`, 估计还是需要指定 `placeholder`

```python
import paddle
import paddle.fluid as fluid
import paddle.dataset.mnist as mnist

def network(image, label):
  # 用户自定义网络，此处以softmax回归为例
    predict = fluid.layers.fc(input=image, size=10, act='softmax')
    return fluid.layers.cross_entropy(input=predict, label=label)

# 定义 py_reader, 指定要么形状, 要么 feed_data_list, 可能还是需要 placehodler的.
reader = fluid.layers.py_reader(capacity=64,
                                shapes=[(-1,1, 28, 28), (-1,1)],
                                dtypes=['float32', 'int64'])
# 和真实的 数据绑定起来.
reader.decorate_paddle_reader(
    paddle.reader.shuffle(paddle.batch(mnist.train(), batch_size=5),
                          buf_size=1000))

# 图上加op. 开始搞
img, label = fluid.layers.read_file(reader)
loss = network(img, label) # 一些网络定义

fluid.Executor(fluid.CUDAPlace(0)).run(fluid.default_startup_program())
exe = fluid.ParallelExecutor(use_cuda=True, loss_name=loss.name)
for epoch_id in range(10):
    reader.start()
        try:
            while True:
                exe.run(fetch_list=[loss.name])
        except fluid.core.EOFException:
            reader.reset()

fluid.io.save_inference_model(dirname='./model',
                              feeded_var_names=[img.name, label.name],
                              target_vars=[loss],
                              executor=fluid.Executor(fluid.CUDAPlace(0)))
```

4. `DataFeeder`

```python
import numpy as np
import paddle
import paddle.fluid as fluid

place = fluid.CPUPlace()

def reader():
    yield [np.random.random([4]).astype('float32'), np.random.random([3]).astype('float32')],

main_program = fluid.Program()
startup_program = fluid.Program()

with fluid.program_guard(main_program, startup_program):
      data_1 = fluid.layers.data(name='data_1', shape=[1, 2, 2])
      data_2 = fluid.layers.data(name='data_2', shape=[1, 1, 3])
      out = fluid.layers.fc(input=[data_1, data_2], size=2)
      # ...

feeder = fluid.DataFeeder([data_1, data_2], place)
reader = feeder.decorate_reader(
      paddle.batch(paddle.dataset.flowers.train(), batch_size=16), multi_devices=False)

exe = fluid.Executor(place)
exe.run(startup_program)
for data in reader():
    outs = exe.run(program=main_program,
                   feed=feeder.feed(data),
                   fetch_list=[out]))
```

