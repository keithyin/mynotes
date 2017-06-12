# caffe2 基本概念入门

理解几个重要概念



## Workspace

* `workspace`:  存放数据的地方。网络中的所有数据都存放在`workspace`中，同时，`workspace`还管理它的`Net`(计算图)。`caffe2` `python`接口，有个默认的`workspace`，名字叫`default`

* 在`caffe2`中，数据由`Blob`保存。`Blob`中保存的数据就是个`ndarray`

  ```python
  from caffe2.python import workspace, model_helper
  import numpy as np
  print("current workspace", workspace.CurrnetWorkspace())
  #打印出当前workspace中的所有 Blob
  print("Current blobs in the workspace: {}".format(workspace.Blobs()))
  #判断当前workspace中 是否存在某 Blob
  print("Workspace has blob 'X'? {}".format(workspace.HasBlob("X")))
  ```

  ​

* 如何向`workspace`中`feed`和 `fetch`数据

  ```python
  X = np.random.randn(2, 3).astype(np.float32)
  print("Generated X from numpy:\n{}".format(X))
  workspace.FeedBlob("X", X) #向workspace中feed数据
  workspace.FetchBlob("X") #从workspace中 Fetch数据
  ```





* 如何切换`workspace`

  ```python
  #打印出当前workspace的名字
  print("Current workspace: {}".format(workspace.CurrentWorkspace()))

  #切换workspace，第二个参数True表示，如果workspace不存在，就创建一个
  workspace.SwitchWorkspace("gutentag", True)

  #再打印一下当前 workspace看看又没啥变化
  print("Current workspace: {}".format(workspace.CurrentWorkspace()))

  ```



## Operators

记住：当用`python`接口创建一个`operator`，其实只是创建了一个`proto buffer`，它指定了` operator`应该是啥。在真正运行的时候，`proto buffer`会被传送给`c++`端计算。

* 如何创建Operator

  ```python
  from caffe2.python import core
  #创建op
  op = core.CreateOperator(
      "Relu", # The type of operator that we want to run
      ["X"], # A list of input blobs by their names
      ["Y"], # A list of output blobs by their names
  )
  #需要注意的是，正常写代码时不用这个麻烦，python接口提供了简单的创建op的方法
  ```



* 如何运行这个`op`

  ```python
  workspace.FeedBlob("X", np.random.randn(2, 3).astype(np.float32))
  workspace.RunOperatorOnce(op) #运行完的值的也会保存在workspace的Blob中。
  ```

  ​

## Nets

`Nets` 就是计算图啦。

```python
from caffe2.python import core
net = core.Net("my_first_net") #创建net
print("Current network proto:\n\n{}".format(net.Proto())) #打印出proto buffer信息

#像net中添加op
X = net.GaussianFill([], ["X"], mean=0.0, std=1.0, shape=[2, 3], run_once=0)
#X是一个BlobReference，保存了它的名字，什么net创造了它，有了它，就不用整天玩string了
#等价于
op = core.CreateOperator("SomeOp", ...)
net.Proto().op.append(op)

W = net.GaussianFill([], ["W"], mean=0.0, std=1.0, shape=[5, 3], run_once=0)
b = net.ConstantFill([], ["b"], shape=[5,], value=1.0, run_once=0)
Y = X.FC([W, b], ["Y"])
#等价于 Y = net.FC([X, W, b], ["Y"]) 使用X.调用op，可以不用传第一个参数。
```



## 打印出创建的计算图

```python
from caffe2.python import net_drawer
from IPython import display
graph = net_drawer.GetPydotGraph(net, rankdir="LR")
display.Image(graph.create_png(), width=800)
```



## 总结

```python
from caffe2.python import workspace, model_helper
import numpy as np

# Create the input data
data = np.random.rand(16, 100).astype(np.float32)

# Create labels for the data as integers [0, 9].
label = (np.random.rand(16) * 10).astype(np.int32)

workspace.FeedBlob("data", data)
workspace.FeedBlob("label", label)

# Create net using a model helper，使用helper创建网络就简单的多了
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

# 初始化参数，初始化参数的图，run一次就好
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

