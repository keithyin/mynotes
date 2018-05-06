# tensorflow 自定义 op

本文只是简单的翻译了 [https://www.tensorflow.org/extend/adding_an_op](https://www.tensorflow.org/extend/adding_an_op) 的简单部分，高级部分请移步官网。



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

在定义 `op`接口 的时候，你需要指定：

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

  // 这里定义的接口 会决定 python中调用 op 时的参数。
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

[请移步官网](https://www.tensorflow.org/extend/adding_an_op)

### **GPU kernels**

[请移步官网](https://www.tensorflow.org/extend/adding_an_op)

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

`Tensorflow` 的 python 接口提供了 `tf.load_op_library` 函数用来加载动态 `library`，同时将 `op` 注册到`tensorflow` 框架上。`load_op_library` 返回一个 `python module`，它包含了 `op`和 `kernel` 的 `python wrapper` 。因此，一旦你编译好了一个 `op`，就可以使用下列代码通过 `python`来执行它：

```python
import tensorflow as tf
zero_out_module = tf.load_op_library('./zero_out.so')
with tf.Session(''):
  zero_out_module.zero_out([[1, 2], [3, 4]]).eval()

# Prints
array([[1, 0], [0, 0]], dtype=int32)
```



记住：生成的函数的名字是 `snake_case` name。如果在`c++`文件中， `op` 的名字是` ZeroOut`，那么在`python` 中，名字是 `zero_out`。

[完整的代码在文章的最后](#代码)



## Verify that the op works

一个验证你的自定义的`op`是否正确工作的一个好的方法是 为它写一个测试文件。创建一个 `zero_out_op_test.py` 文件，内容为：

```python
import tensorflow as tf

class ZeroOutTest(tf.test.TestCase):
  def testZeroOut(self):
    zero_out_module = tf.load_op_library('./zero_out.so')
    with self.test_session():
      result = zero_out_module.zero_out([5, 4, 3, 2, 1])
      self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

if __name__ == "__main__":
  tf.test.main()
```

然后运行这个 `test`





## 代码

```c++
//zero_out.cc 文件
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // 将输入 tensor 从 context 中取出。
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // 创建一个 ouput_tensor, 使用 context->allocate_ouput() 给它分配空间。
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);
  }
};
REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
```

```shell
#创建动态链接库的命令
g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I $TF_INC -O2
```



## 注意事项

* `REGISTER_OP("ZeroOut")` 中， op的名字必须遵循驼峰命名法。
* `Compute()` 方法一定要线程安全。
* 定义`OP` 的类名可以随便起，但也不要太随便。在 `REGISTER_KERNEL_BUILDER` 时对应好就可以了。
* 在`C++` 文件中，如果 定义`op`接口时的名字是 `ZeroOut` 的话，那么，在`python` 中的 `op`名字是`zero_out`。




## 定义梯度

**方法一：** 在python 中用 tf api实现

```python
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

@ops.RegisterGradient("ZeroOut")
def _zero_out_grad(op, grad):
  """The gradients for `zero_out`.
  Args:
    op: The `zero_out` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `zero_out` op.

  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  to_zero = op.inputs[0]
  shape = array_ops.shape(to_zero)
  index = array_ops.zeros_like(shape)
  # 从这看出，计算梯度的时候，也是喜欢把输入进来的梯度 展平。
  first_grad = array_ops.reshape(grad, [-1])[0] 
  to_zero_grad = sparse_ops.sparse_to_dense([index], shape, first_grad, 0)
  return [to_zero_grad]  # List of one Tensor, since we have one input
```



**方法二：** 使用 c++ 定义一个 GradOp，然后在 python 中调用这个 Op

```python
import tensorflow as tf
from tensorflow.python.framework import ops
import roi_pooling_op
import pdb


@ops.RegisterShape("RoiPool")
def _roi_pool_shape(op):
  """Shape function for the RoiPool op.
  """
  dims_data = op.inputs[0].get_shape().as_list()
  channels = dims_data[3]
  dims_rois = op.inputs[1].get_shape().as_list()
  num_rois = dims_rois[0]

  pooled_height = op.get_attr('pooled_height')
  pooled_width = op.get_attr('pooled_width')

  output_shape = tf.TensorShape([num_rois, pooled_height, pooled_width, channels])
  return [output_shape, output_shape]

@ops.RegisterGradient("RoiPool")
def _roi_pool_grad(op, grad, _):
  """The gradients for `roi_pool`.
  Args:
    op: The `roi_pool` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `roi_pool` op.
  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  data = op.inputs[0]
  rois = op.inputs[1]
  argmax = op.outputs[1]
  pooled_height = op.get_attr('pooled_height')
  pooled_width = op.get_attr('pooled_width')
  spatial_scale = op.get_attr('spatial_scale')

  # compute gradient, roi_pool_grad 是在c++ 中定义过的op
  data_grad = roi_pooling_op.roi_pool_grad(data, rois, argmax, grad, pooled_height, pooled_width, spatial_scale)

  return [data_grad, None]  # List of one Tensor, since we have one input
```

**方法三：** 在 c++ 代码中 REGISTER_OP_GRADIENT， 这个方法不需要我们在python中做其它额外操作。

* 首先，熟悉一下 FunctionDefHelper，tf中的大多数 Grad 都是通过 FDH 定义的，下面的MatMulGrad 就是个例子

  ```c++
  // 地址 https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/math_grad.cc
  #include <vector>
  #include "tensorflow/core/framework/function.h"
  #include "tensorflow/core/lib/core/errors.h"

  namespace tensorflow {

  typedef FunctionDefHelper FDH;
  static Status MatMulGradHelper(FunctionDef* g, const string& opname,
                                 const string& attr_adj_x,
                                 const string& attr_adj_y, const string& x0,
                                 bool ax0, const string& x1, bool ax1,
                                 const string& y0, bool ay0, const string& y1,
                                 bool ay1) {
    *g = FDH::Define(
        // Arg defs，输入参数定义
        {"x: T", "y: T", "dz: T"},
        // Ret val defs ，返回值定义
        {"dx: T", "dy: T"},
        // Attr defs，属性定义
        {{"T: {half, float, double}"}},
        // Nodes，节点定义，梯度计算可能会需要多个 op
        {
            { // 第一个 op，用来求 dx
              {"dx"}, //op 的输出
              opname, // op 的名字
              {x0, x1}, // op 的输入
              { //Attr
                {"T", "$T"}, // op 的属性， $T 代表参数定义/属性定义时的 T
                {attr_adj_x, ax0}, // op 的属性 
                {attr_adj_y, ax1} // op的属性
              }
            },
            {
              {"dy"},
              opname,
              {y0, y1},
              {
                {"T", "$T"}, 
                {attr_adj_x, ay0}, 
                {attr_adj_y, ay1}
              }
            },
        });
    return Status::OK();
  }

  Status MatMulGradCommon(const string& opname, const string& attr_adj_x,
                          const string& attr_adj_y, const AttrSlice& attrs,
                          FunctionDef* g) {
    DataType T;
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "T", &T));
    if (T == DT_COMPLEX64 || T == DT_COMPLEX128) {
      return errors::Unimplemented(
          "MatMul gradient for complex is not supported yet.");
    }
    bool ta;
    bool tb;
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, attr_adj_x, &ta));
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, attr_adj_y, &tb));
    if (!ta && !tb) {
      return MatMulGradHelper(g, opname, attr_adj_x, attr_adj_y, "dz", false, "y",
                              true, "x", true, "dz", false);
    }
    if (!ta && tb) {
      return MatMulGradHelper(g, opname, attr_adj_x, attr_adj_y, "dz", false, "y",
                              false, "dz", true, "x", false);
    }
    if (ta && !tb) {
      return MatMulGradHelper(g, opname, attr_adj_x, attr_adj_y, "y", false, "dz",
                              true, "x", false, "dz", false);
    }
    CHECK(ta && tb);
    return MatMulGradHelper(g, opname, attr_adj_x, attr_adj_y, "y", true, "dz",
                            true, "dz", true, "x", true);
  }

  Status MatMulGrad(const AttrSlice& attrs, FunctionDef* g) {
    return MatMulGradCommon("MatMul", "transpose_a", "transpose_b", attrs, g);
  }
  // Grad 函数的签名 (const AttrSlice& attrs, FunctionDef* g)
  // 第一个用来接受 所求梯度的 op 的属性，第二个用来定义 Grad 计算操作。
  REGISTER_OP_GRADIENT("MatMul", MatMulGrad);  
  ```

  > 可以看出，Grad 的定义就是使用 已有的 op 来定义 Grad 的计算。



## 总结

`tensorflow` 自定义 `op` 的方法可以总结为：

1. 写个 `diy_op.cc` 文件
2. 用 `g++` 把这个文件编译成动态链接库
3. 在 `python` 中使用 `tf.load_op_library`  将库导入。
4. 就可以使用了。

其它的方法就是用 `bazel` 编译了，毕竟用的不多。



## 参考资料

[https://www.tensorflow.org/extend/adding_an_op](https://www.tensorflow.org/extend/adding_an_op)

