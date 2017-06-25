# tensorflow 自定义 op

可能需要新定义 `c++ operation` 的几种情况：

* 现有的 `operation` 组合不出来你想要的 `op`
* 现有的 `operation` 组合 出来的 `operation` 十分低效
* 如果你想要手动融合一些操作。



为了实现你的自定义操作，你需要做一下几件事：

1. 在 c++ 文件中注册一个新`op`： `Op registration` 定义了 `op` 的功能接口，它和 `op` 的实现是独立的。例如：`op registration` 定义了 `op` 的名字和 `op`的输出输出。它同时也定义了 `shape` 方法，被用于 `tensor` 的 `shape` 接口。 
2. 在 `c++` 中实现 `op`：`op` 的实现称之为 `kernel` ，它是`op` 的一个具体实现。对于不同的输入输出类型或者 架构（CPUs，GPUs）可以有不同的 `kernel` 实现 。
3. 创建一个 `python wrapper`（可选的）： 这个 `wrapper` 是一个 公开的 `API`，用来在 `python`中创建 `op`。 `op registration` 会生成一个默认的 `wrapper`，我们可以直接使用或者自己添加一个。
4. 写一个计算 `op` 梯度的方法（可选）。
5. 测试 `op`：为了方便，我们通常在 `python` 中测试 `op`，但是你也可以在 `c++` 中进行测试。如果你定义了 `gradients`，你可以 通过 `Python` 的  [gradient checker](https://www.tensorflow.org/api_docs/python/tf/test/compute_gradient_error) 验证他们。  这里有个例子[`relu_op_test.py`](https://www.github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/kernel_tests/relu_op_test.py) ，测试 `ReLU-like` 的 `op` 的 前向和梯度过程。



## Define the op's interface

**You define the interface of an op by registering it with the TensorFlow system. **



在注册 `op` 的时候，你需要指定：

*  `op` 的名字
*  `op` 的输入（名字，类型），`op` 的输出（名字，类型）
*  `docstrings`  
*  `op` 可能需要的 一些  [attrs](https://www.tensorflow.org/extend/adding_an_op#attrs) 



**为了演示这个到底怎么工作的，我们来看一个简单的例子：**

* 定义一个 `op` ：输入是一个 `int32` 的 `tensor` ，输出是输入的 拷贝，除了第一个元素保留，其它全都置零。



为了创建这个 `op` 的接口， 我们需要：

* 创建一个文件，名字为  `zero_out.cc`.  然后调用 `REGISTER_OP` 宏，这个宏定义了 你的 `op` 的接口 ：

  ```c++
  #include "tensorflow/core/framework/op.h"
  #include "tensorflow/core/framework/shape_inference.h"

  using namespace tensorflow;

  REGISTER_OP("ZeroOut")
      .Input("to_zero: int32")
      .Output("zeroed: int32")
      .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
      });
  ```

  ​

这个 `ZeroOut op` 接收一个 `int 32` 的 `tensor` 作为输入，输出同样也是一个 `int32`的 `tensor`。这个 `op` 也使用了一个 `shape` 方法来确保输入和输出的维度是一样的。例如，如果输入的`tensor` 的shape 是 `[10, 20]`，那么，这个 `shape` 方法保证输出的 `shape` 也是 `[10, 20]`。 

> 注意： op 的名字必须遵循驼峰命名法，而且要保证 op 的名字的唯一性。



## Implement the kernel for the op

当你 定义了 `op` 的接口之后，你可以提供一个或多个 关于`op` 的实现。

为了实现这些 `kernels`：

* 创建一个类，继承 `OpKernel` 类
* 重写 `OpKernel` 类的 `Compute` 方法
  * `Compute` 方法提供了一个 类型为 `OpKernelContext* ` 的`context` 参数 ，从这里，我们可以访问到一些有用的信息，比如 输入 和 输出 `tensor`

将 `kernel` 代码也放到 之前创建的 `zero_out.cc` 文件中：

```c++
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // 获取输入 tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // 创建输出 tensor, context->allocate_output 用来分配输出内存？
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // 执行计算操作。
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);
  }
};
```



在实现了 `kernel` 之后，就可以将这个注册到 `tensorflow` 系统中去了。在注册时，你需要对 `op` 的运行环境指定一些限制。例如，你可能有一个 `kernel` 代码是给 `CPU` 用的，另一个是给 `GPU`用的。通过把下列代码添加到 `zero_out.cc` 中来完成这个功能：

```c++
REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
```

> 注意：你实现的 `OpKernel` 的实例可能会被并行访问，所以，请确保 `Compute`方法是线程安全的。保证访问 类成员的 方法都加上 mutex。或者更好的选择是，不要通过 类成员来分享 状态。考虑使用 [`ResourceMgr`](https://www.github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/framework/resource_mgr.h) 来追踪状态。



### **Multi-threaded CPU kernels**



### **GPU kernels**



## Build the op library

**使用系统编译器 编译 定义的 `op`**

我们可以使用 系统上的 `c++` 编译器 `g++` 或者 `clang` 来编译 `zero_out.cc` 。二进制的 `PIP 包` 已经将编译所需的 头文件 和 库 安装到了系统上。`Tensorflow` 的 `python library` 提供了一个用来获取 头文件目录的函数 `get_include`。下面是这个函数在 `ubuntu` 上的输出：

```shell
$ python
>>> import tensorflow as tf
>>> tf.sysconfig.get_include()
'/usr/local/lib/python2.7/site-packages/tensorflow/include'
```

 假设你已经安装好了 `g++` ，你可以使用 下面一系列的命令 将你的 `op` 编译成一个 动态库。

```shell
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -I $TF_INC -O2
```

> Note on `gcc` version `>=5`: gcc uses the new C++ [ABI](https://gcc.gnu.org/gcc-5/changes.html#libstdcxx) since version `5`. The binary pip packages available on the TensorFlow website are built with `gcc4` that uses the older ABI. If you compile your op library with `gcc>=5`, add `-D_GLIBCXX_USE_CXX11_ABI=0` to the command line to make the library compatible with the older abi. Furthermore if you are using TensorFlow package created from source remember to add `--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"` as bazel command to compile the Python package.



## Use the op in Python



## 参考资料

[https://www.tensorflow.org/extend/adding_an_op](https://www.tensorflow.org/extend/adding_an_op)

