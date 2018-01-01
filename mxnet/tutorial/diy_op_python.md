# mxnet: 自定义 operation （Python）

**python 端自定义 operation 的步骤总结为**

* 继承 `mx.operator.CustomOp` 然后重写 `forward` 和 `backward` 方法
* 继承 `mx.operator.CustomOpProp` 并注册 `op`
* 用的时候 使用  `mx.sym.Custom`