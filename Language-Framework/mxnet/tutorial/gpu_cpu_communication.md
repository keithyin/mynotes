# mxnet cpu gpu 通信

* cpu 和 gpu 上的 数据不能直接运算
* 但是, 可以赋值

```python
import mxnet as mx
v_gpu = mx.nd.zeros(shape=(100, 100), ctx=mx.gpu())
v_cpu = mx.nd.ones(shape=(100, 100), ctx=mx.cpu())
v_gpu[:] = v_cpu
```

