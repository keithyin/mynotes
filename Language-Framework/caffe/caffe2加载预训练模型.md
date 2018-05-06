# caffe2 加载预训练模型

利用 [Model Zoo](https://github.com/caffe2/caffe2/wiki/Model-Zoo), 取一些训练好的模型用来测试. 在 [Model Zoo](https://github.com/caffe2/caffe2/wiki/Model-Zoo)中, 你可以看到不同的模型,它们已经在相关数据集上被训练好, 下面将展示 如何使用这些预训练模型的基本步骤.


## Model Download Options

你可以通过 `Caffe2` 的 `models.download` 模块 从[Github caffe2/models](http://github.com/caffe2/models) 中去获取 预训练好的 模型. `caffe2.python.models.download` 需要一个模型的名称作为其参数. 在仓库中找到你想下载模型的名字, 然后传给 `caffe2.python.models.download` 作为参数就 OK 了.
例子:

```shell
python -m caffe2.python.models.download -i squeezenet
```
如果下载顺利, 你就会得到一个 `squeezenet` 的副本, 它放在 模型文件夹下, 如果使用了 `-i` 标签, 那么模型会被安装在本机的 `/caffe2/python/models` 文件夹下. 当然, 你也可以 `clone` 所有的模型: `git clone https://github.com/caffe2/models`


## Overview

在此教程汇中, 我们将会使用 `squeezenet` 模型去鉴别图象中的物体. 如果你已经看过 `Image Pre-Processing`教程, 你会发现, 我们使用 `rescale` 和 `crop` 函数去准备图片, 然后 将图片格式化成 `CHW, BGR`, 最终是 `NCHW`. 我们也修正了图象均值, 可以使用计算好的均值, 也可以直接使用 128作为均值.

你会发现, 加载预训练的模型是非常简单的, 只需要短短几行代码就可以完成:

```python
# 1. 读 protobuf 文件

with open("init_net.pb") as f:
     init_net = f.read()
with open("predict_net.pb") as f:
     predict_net = f.read()   

# 2. 使用workspace中的Predictor函数加载protobufs中的blobs.
p = workspace.Predictor(init_net, predict_net)

# 3.运行 net, 然后得到结果
results = p.run([img])

```

返回的结果是一个多维的概率数组, 每行表示了属于某类的概率.
