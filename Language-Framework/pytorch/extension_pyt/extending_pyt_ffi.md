# pytorch学习笔记（十八）：扩展 pytorch-ffi

上篇博文已经介绍了如何通过 继承 `Function` ，然后使用`python` 来扩展 `pytorch`， 本文主要介绍如何通过 `cffi` 来扩展 `pytorch` 。



官网给出了一个 `MyAdd` 的 `Demo` [github地址](https://github.com/pytorch/extension-ffi)，本文通过 这个 `Demo` 来搞定如何 通过 `cffi` 来扩展 `pytorch`。



**从github上clone下来代码，目录结构是这样的**

* package：
* script：（这个部分的示例 是 扩展包 仅当前 项目可见。）




## 自定义 OP

`pytorch` 自定义 `op` 的基本步骤总结如下。

**一、C部分**：

- `new_op.h` :   `forward(), backward()` 接口声明
- `new_op.c`： 实现 `forward(), backward()` `CPU` 代码 
- `new_op.cu`： 实现 `forward(), backward()` `GPU` 代码



**二、编译上面写的 C/CUDA 代码**



**三、python部分：**

* 用 `Function` 包装 `C OP`
* 用 `Module` 包装 `Function`



下面，再来看一下 官方的 `Demo`

## 再看Script 部分

`Script` 部分的文件结构如下：

* `src/` ： 放着 C 代码
* `functions/` ： `Function` 包装
* `modules/` ： `Module` 包装
* `build` ： 编译 `C` 源码的 代码



**C/CUDA 代码**

```c
#include <TH/TH.h>

int my_lib_add_forward(THFloatTensor *input1, THFloatTensor *input2,
		       THFloatTensor *output)
{
  if (!THFloatTensor_isSameSizeAs(input1, input2))
    return 0;
  THFloatTensor_resizeAs(output, input1);
  THFloatTensor_cadd(output, input1, 1.0, input2);
  return 1;
}

int my_lib_add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
{
  THFloatTensor_resizeAs(grad_input, grad_output);
  THFloatTensor_fill(grad_input, 1);
  return 1;
}
```



**编译用代码**

```python
import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['src/my_lib.c']
headers = ['src/my_lib.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/my_lib_cuda.c']
    headers += ['src/my_lib_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

ffi = create_extension(
    '_ext.my_lib', # _ext/my_lib 编译后的动态 链接库 存放路径。
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda
)

if __name__ == '__main__':
    ffi.build()
```



**Function Wrapper**

```python
import torch
from torch.autograd import Function
from _ext import my_lib
from torch.autograd import Variable


class MyAddFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2):
        output = input1.new()
        if not input1.is_cuda:
            my_lib.my_lib_add_forward(input1, input2, output)
        else:
            my_lib.my_lib_add_forward_cuda(input1, input2, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        t_grad_output = grad_output.data
        t_grad_input = t_grad_output.new().resize_as_(t_grad_output).zero_()
        grad_input = Variable(t_grad_input, requires_grad=grad_output.requires_grad, volatile=grad_output.volatile)
        if not grad_output.is_cuda:
            my_lib.my_lib_add_backward(grad_output.data, t_grad_input)
        else:
            my_lib.my_lib_add_backward_cuda(grad_output.data, t_grad_input)
        return grad_input, grad_input
```



**Module Wrapper**

```python
class MyAddModule(Module):
    def forward(self, input1, input2):
        return MyAddFunction.apply(input1, input2)
```





