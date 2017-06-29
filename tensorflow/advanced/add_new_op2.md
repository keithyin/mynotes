# tensorflow 自定义 op：2

上一篇文章翻译了 `tensorflow` 自定义 `op` 的简单操作。这篇开始向高级进发了



## Building advanced features into your op

这里会讨论一些比较复杂的事情，他们包括：

- [Conditional checks and validation](https://www.tensorflow.org/extend/adding_an_op#validate)
- Op registration
  - [Attrs](https://www.tensorflow.org/extend/adding_an_op#attrs)
  - [Attr types](https://www.tensorflow.org/extend/adding_an_op#attr_types)
  - [Polymorphism](https://www.tensorflow.org/extend/adding_an_op#polymorphism)
  - [Inputs and outputs](https://www.tensorflow.org/extend/adding_an_op#inputs_outputs)
  - [Backwards compatibility](https://www.tensorflow.org/extend/adding_an_op#backward_compat)
- GPU 支持
  - [Compiling the kernel for the GPU device](https://www.tensorflow.org/extend/adding_an_op#compiling_kernel)
- [Implement the gradient in Python](https://www.tensorflow.org/extend/adding_an_op#implement_gradient)
- [Shape functions in C++](https://www.tensorflow.org/extend/adding_an_op#shape_functions)

## 条件检查 和 验证

之前的例子假定 自定义的 `op` 可以用在任何形状的 `tensor` 上。如果只要求用在 `vector`上呢？这就意味着 对上面实现的 `OpKernel` 添加一个检查。

```c++
void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor.shape()),
                errors::InvalidArgument("ZeroOut expects a 1-D vector."));
    // ...
  }
```

这段代码 断言 输入 `tensor` 是个 `vector` ，如果输入不是 `vector`，就会返回一个 `InvalidArgument` 状态。

**[`OP_REQUIRES` 宏](https://www.github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/lib/core/errors.h) 有三个参数：**

* `context`:上下文，可以是 `OpKernelContext` 指针，也可以是`OpKernelConstruction` 指针。主要是用`context`的 `SetStatus`方法。
* `condition`: 例如，[这里](https://www.github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/framework/tensor_shape.h)有一些验证 `tensor shape`的 函数。 
* `error`: 表示 如果 条件不满足，报什么错误。它是一个 `Status` 对象。见 [`tensorflow/core/lib/core/status.h`](https://www.github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/lib/core/status.h)。 `Status` 包含 类型（通常是 `InvalidArgument` ）和 消息。创建 `error` 的函数在   [`tensorflow/core/lib/core/errors.h`](https://www.github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/lib/core/errors.h)。

如果你想测试 某个函数返回的 `Status` 是不是一个 `error`，使用 [`OP_REQUIRES_OK`](https://www.github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/lib/core/errors.h)。 



## Op registration（注册 Op）

### Attrs

一个使用了 `attr` 的例子：

```c++
REGISTER_OP("ZeroOut")
    .Attr("preserve_index: int")
    .Input("to_zero: int32")
    .Output("zeroed: int32");
```



`Ops` 可以拥有属性，当`Op` 被添加到图中去的时候，这些属性已被指定。属性是用来配置 `op` 的，属性的值可以在 `kernel` 实现的时候访问到，也可以在 `op`注册中的输入输出类型中访问。

* 能用 `Input` 的情况就尽量用 `Input`
* 因为 `Attr`是常量，必须在建图的时候就被指定
* 相反，`Inputs` 是 `Tensors`，它的值可以是动态的。
* `Inputs` 的值可以每一步都不同，可以用 `feed`指定。 
* `Attrs` 被用来 做 `Inputs`做不了的事情，比如：任何可以改变签名的设置（数量，数量，或者输出输出的类型），这些不能在每一步都改变。



在注册 `Op` 的时候定义 `Attr`，通过 `Attr` 方法来指定 它的名字，期望是下面这种形式：

```c++
<name>: <attr-type-expr>
```

* name 以 字母开头，可以使用下划线和数字



例如，如果你想要 `ZeroOp` 保存用户指定的 `index`，而不是只使用`index 0`,你可以这样注册 `Op`:

```c++
REGISTER_OP("ZeroOut")
    .Attr("preserve_index: int")
    .Input("to_zero: int32")
    .Output("zeroed: int32");
```

* 这里要注意的是：[attr 的类型](https://www.tensorflow.org/extend/adding_an_op#attr_types) 和 用于输入输出 的 [tensor 的类型](https://www.tensorflow.org/programmers_guide/dims_types)是不同的。



然后，在实现 `kernel` 的时候就可以 通过 `context` 访问这些属性：

```c++
class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the index of the value to preserve
    OP_REQUIRES_OK(context,
                   context->GetAttr("preserve_index", &preserve_index_));
    // Check that preserve_index is positive
    OP_REQUIRES(context, preserve_index_ >= 0,
                errors::InvalidArgument("Need preserve_index >= 0, got ",
                                        preserve_index_));
  }
  void Compute(OpKernelContext* context) override {
    // ...
  }
 private:
  int preserve_index_;
};
```



然后属性在 `Compute` 中使用：

```c++
void Compute(OpKernelContext* context) override {
    // ...


    // We're using saved attr to validate potentially dynamic input
    // So we check that preserve_index is in range
    OP_REQUIRES(context, preserve_index_ < input.dimension(0),
                errors::InvalidArgument("preserve_index out of range"));

    // Set all the elements of the output tensor to 0
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the requested input value
    output_flat(preserve_index_) = input(preserve_index_);
  }
```

### 属性类型 ： Attr types

属性支持下列类型：

- `string`: Any sequence of bytes (not required to be UTF8).
- `int`: 有符号整型
- `float`: 浮点型
- `bool`: 布尔型
- `type`: One of the (non-ref) values of [`DataType`](https://www.github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/framework/types.cc).
- `shape`: A [`TensorShapeProto`](https://www.github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/framework/tensor_shape.proto).
- `tensor`: A [`TensorProto`](https://www.github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/framework/tensor.proto).
- `list(<type>)`: A list of `<type>`, where `<type>` is one of the above types. Note that `list(list(<type>))` is invalid.

See also: [`op_def_builder.cc:FinalizeAttr`](https://www.github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/framework/op_def_builder.cc) for a definitive list.



**默认值和限制：**

* `Attrs` 可能有 默认值，有些类型的属性 也可能有限制。为了定义一个有限制的属性，你可以使用以下方法来指定 `<arrt-type-expr>`：

  ```c++
  // 1.  {'<string1>', '<string2>'}， 这样，属性的值只能取 'string1' 或 'string2'
  REGISTER_OP("EnumExample")
      .Attr("e: {'apple', 'orange'}");

  // 2.  {<type1>, <type2>}， 这样的话，属性值可以是 'type1' 或 'type2'。
  // t 是个 type 类型！！！{<type1>, <type2>} 表示 了 t 是 type 类型。
  REGISTER_OP("RestrictedTypeExample")
      .Attr("t: {int32, float, bool}");
  ```

  ```c++
  // 这种语法表示的是，a是个 int，而且值要大于等于2
  REGISTER_OP("MinIntExample")
      .Attr("a: int >= 2");
  ```

  ```c++
  // 设置 默认值
  REGISTER_OP("AttrDefaultExample")
      .Attr("i: int = 0");
  ```




### 多态性

**类型多态性：**

```c++
REGISTER_OP("ZeroOut")
    .Attr("T: {float, int32}")
    .Input("to_zero: T")
    .Output("zeroed: T");
```







## 注意

`tensorflow` 自定义 `op` 需要注意的事：

* ​


## 参考资料

[https://www.tensorflow.org/extend/adding_an_op](https://www.tensorflow.org/extend/adding_an_op)

