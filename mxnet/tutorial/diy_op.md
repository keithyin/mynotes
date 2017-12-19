# mxnet : 自定义 op

创建一个 `op` ，一般情况下需要创建 三个文件：

* `new_op.h` ： 头文件
* `new_op.cc` ：  CPU 代码
* `new_op.cu`：  GPU 代码



`mxnet` 将一个 `OP` 分解成了三个部分：

* `op` 参数部分： 有些 `op` 会有自己的一些参数，比如 `max_pool`, `stride` 就是其 参数，这个参数值，在创建这个 `op` 时就已经确定，不会再改变！！！！
  * 继承 `dmlc.Parameter`
* `op` 计算部分： `op` 的核心计算部分
  * 继承 `Operator`
* `op` 属性部分： 用于 `infer_shape, infer_type` 
  * 继承 `OperatorProperty`



一个二次方程的例子：$f(x)=ax^2+bx+c$

## Parameter

`a,b,c` 为 `op` 的 `Parameter`

```c++
// new_op.h
struct QuadraticParam : public dmlc::Parameter<QuadraticParam> {
  float a, b, c; // 变量声明
  // 用宏搞定剩余的
  DMLC_DECLARE_PARAMETER(QuadraticParam) {
    DMLC_DECLARE_FIELD(a)
      .set_default(0.0)
      .describe("Coefficient of the quadratic term in the quadratic function.");
    DMLC_DECLARE_FIELD(b)
      .set_default(0.0)
      .describe("Coefficient of the linear term in the quadratic function.");
    DMLC_DECLARE_FIELD(c)
      .set_default(0.0)
      .describe("Constant term in the quadratic function.");
  }
};
```



## Property





