# caffe2入门

`caffe2`中，我们可以使用`python`接口定义网络.



## Blob Tensor Workspace

* `Blob`： caffe中的保存数据的单元叫做`Blob`. `Blob`是内存中的一个命名单元。
  * 大多数的`Blob`中包含着一个`Tensor`  （在`Python`接口中，`Tensor`就是`numpy`的基本类型`ndarray`）

* `Workspace`：`Workspace`中保存着所有的`Blob`.

  * 如何将`Blob` `feed`进 `workspace` ，然后获取`Blob`

    ```python
    from caffe2.python import workspace, model_helper
    import numpy as np
    # Create random tensor of three dimensions
    x = np.random.rand(4, 3, 2)
    print(x)
    print(x.shape)
    workspace.FeedBlob("my_x", x)
    x2 = workspace.FetchBlob("my_x")
    print(x2)
    ```

## Nets, Operators

* `Nets` :可以看做一个`DAG`，其中，`operator`作为`Graph`的节点。`Blob`的流向称为边
* `Operator`: 操作

```python
from caffe2.python import workspace, model_helper
import numpy as np

# Create the input data
data = np.random.rand(16, 100).astype(np.float32)

# Create labels for the data as integers [0, 9].
label = (np.random.rand(16) * 10).astype(np.int32)

workspace.FeedBlob("data", data)
workspace.FeedBlob("label", label)

# Create model using a model helper
m = model_helper.ModelHelper(name="my first net")

# how to init the parameters 
weight = m.param_init_net.XavierFill([], 'fc_w', shape=[10, 100])
bias = m.param_init_net.ConstantFill([], 'fc_b', shape=[10, ])

# define OP
fc_1 = m.net.FC(["data", "fc_w", "fc_b"], "fc1")
pred = m.net.Sigmoid(fc_1, "pred")
[softmax, loss] = m.net.SoftmaxWithLoss([pred, "label"], ["softmax", "loss"])


#图到此已经定义完毕
###############################################################################

# 初始化参数
workspace.RunNetOnce(m.param_init_net)

#创建训练图
workspace.CreateNet(m.net)

#迭代训练
# Run 100 x 10 iterations
for j in range(0, 100):
    data = np.random.rand(16, 100).astype(np.float32)
    label = (np.random.rand(16) * 10).astype(np.int32)

    workspace.FeedBlob("data", data)
    workspace.FeedBlob("label", label)

    workspace.RunNet(m.name, 10)   # run for 10 times
    
    #可以将值取出来看看
    print(workspace.FetchBlob("softmax"))
    print(workspace.FetchBlob("loss"))
```

