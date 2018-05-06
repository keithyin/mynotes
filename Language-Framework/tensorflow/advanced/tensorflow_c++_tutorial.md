# tensorflow c++ tutorial(1)

>  继续当官方文档的搬运工



这段代码放到下载的tensorflow源码的 tensroflow/cc/example下

```c++
// tensorflow/cc/example/example.cc

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

int main() {
  using namespace tensorflow;
  using namespace tensorflow::ops;
  Scope root = Scope::NewRootScope();
  // Matrix A = [3 2; -1 0]
  auto A = Const(root, { {3.f, 2.f}, {-1.f, 0.f}});
  // Vector b = [3 5]
  auto b = Const(root, { {3.f, 5.f}});
  // v = Ab^T
  auto v = MatMul(root.WithOpName("v"), A, b, MatMul::TransposeB(true));
  std::vector<Tensor> outputs;
  ClientSession session(root);
  // Run and fetch v
  TF_CHECK_OK(session.Run({v}, &outputs));
  // Expect outputs[0] == [19; -3]
  LOG(INFO) << outputs[0].matrix<float>();
  return 0;
}
```

**核心概念解析：**



**1. NewRootScope**

> [`tensorflow::Scope`](https://www.tensorflow.org/api_docs/cc/class/tensorflow/scope) is the main data structure that holds the current state of graph construction. 

这个`Scope` 应该就是 `python` 中的 `namescope`。给 `op` 加前缀用的。

* `op`都会以`scope` 作为第一个参数。



*当执行 `Scope root = Scope::NewRootScope()`时，底层实际上执行了一下操作：*

* 创建了一个图，这样 `op` 就能被添加到这个图中去 `Graph`
* 同时也创建了一个 `tensorflow::Status` ，用来指示创建 `op` 的过程中，有没有出现错误。

> The `Scope` class has value semantics, thus, a `Scope` object can be freely copied and passed around.



*如何创建子 `Scope`：*

* `root.NewSubScope()`



*Scope 类控制的一些属性：*

* op 的名字
* Set of control dependencies for an operation
* Device placement for an operation
* Kernel attribute for an operation



**2.  创建 op**

```c++
// Not recommended
MatMul m(scope, a, b);

// Recommended
auto m = MatMul(scope, a, b);
```



**3. 常量**

```c++
auto f = Const(scope, 42.0f);
auto s = Const(scope, "hello world!");

// 2x2 matrix
auto c1 = Const(scope, { {1, 2}, {2, 4}});
// 1x3x1 tensor
auto c2 = Const(scope, { { {1}, {2}, {3}}});
// 1x2x0 tensor
auto c3 = ops::Const(scope, { { {}, {}}});

// 2x2 matrix with all elements = 10
auto c1 = Const(scope, 10, /* shape */ {2, 2});
// 1x3x2x1 tensor
auto c2 = Const(scope, {1, 2, 3, 4, 5, 6}, /* shape */ {1, 3, 2, 1});
```



**4. 执行图**

> 执行图的话，需要session

```c++
Scope root = Scope::NewRootScope();
auto c = Const(root, { {1, 1}});
auto m = MatMul(root, c, { {42}, {1}});

ClientSession session(root);
std::vector<Tensor> outputs; //为什么用 vector 呢？因为可能 fetch 出多个值
session.Run({m}, &outputs);
```



**5. placeholder**

```c++
Scope root = Scope::NewRootScope();
auto a = Placeholder(root, DT_INT32);
// [3 3; 3 3]
auto b = Const(root, 3, {2, 2});
auto c = Add(root, a, b);
ClientSession session(root);
std::vector<Tensor> outputs;

// Feed a <- [1 2; 3 4]
session.Run({ {a, { {1, 2}, {3, 4}}}}, {c}, &outputs);
// outputs[0] == [4 5; 6 7]
```



## 参考资料

[https://www.tensorflow.org/api_guides/cc/guide](https://www.tensorflow.org/api_guides/cc/guide)

[https://www.tensorflow.org/api_docs/cc/class/tensorflow/client-session](https://www.tensorflow.org/api_docs/cc/class/tensorflow/client-session)

