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

