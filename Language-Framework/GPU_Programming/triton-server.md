# Overview
一个通用的 inference server架构一般由两个部分组成：①client，②server

server：
1. k8s cluster
   1. load balancer
   2. model repository persistent volume
   3. metrics service （推理服务监控）
   4. containerized inference service -- triton （这里是triton的位置）
      1. triton只是推理框架中的一部分
      2. 主要是做 单节点上进行推理
      3. 可以是CPU推理，也可以是GPU推理
      4. 模型可以来自多个框架：tensorflow、pytorch、onnx、tensorrt
      5. triton 包含 tensorflow、pytorch、onnx、tensorrt 的推理库

总结：
1. K8s cluster：inference service with multi-node with multi-card (containerized service orchestration toolset)
2. triton: inferce service with one mode with one/multi GPU card (typically within one container as a service)
3. tensorrt: inference acceleration library for nn models

Triton basic functionalities:
1. multiple model frameworks supported (tf, torch, onnx, tensorrt, and custom backends)
2. CPU, GPU, multi-GPU support
3. Concurrent model execution (cpu level optimization)
   1. 一个模型，多个线程进行推理
   2. 多个模型，多个线程进行推理
5. server (http/rest, gRPC apis)
6. intergrates with orchestration systems and auto scalers via latency and health metrics
7. model management, load/unload, model update
8. open source with monthly releases on git and ngc as docker container


# Design Basics of Triton
基于推理的lifecycle来设计triton, 从推理请求开始 到 结束，会有哪些步骤，根据步骤设计系统.

1. multi model frameworks supported -> this is subject to frameworks, should be decoupled by frameworks -> Backends
   1. 基于深度学习框架的，这部分工作是交给深度学习框架的（pytorch，tf，trt）
   2. 通过backends是可以解耦合的
3. common functionalities
   1. common backend management
      1. backend load, etc.
     
   2. model management - load/unload, model update, query model status, etc.
   3. concurrent model execution - instance management
      1. 从高性能角度来说
   5. request queue dispatch and scheduler
      1. 请求队列的调度
   6. infer lifecycle management
      1. inference request management
      2. inference response management （结果打包）
     
   8. GRPC related
      1. grpc server
     
基于 abstraction of customer Scenarios (note: mostly model type relevent):
1. Roughly can be classified into three types
   1. simple independent model  (单一，无依赖的模型）
   2. ensembled with a model pipeline （模型组合，模型运行的pipeline）
   3. stateful model （有状态的模型，语言模型）
  

Common Scenario 1：One API using multiple copies of the same model on a GPU
* 一个GPU上启动 多个 模型实例（多个线程进行推理！）。这样就可以就可以同时处理多个请求！
* 单模型，多线程
* request 的 scheduler 进行调度

Common Scenario 2：Many APIs using multiple different models on a GPU
* 多模型，多线程。
* request 的 scheduler 进行调度

Model type is the key, you will find triton doc is **not** so difficult to read
* Stateless
  * CV
  * default scheduler, dynamic batch。
    * default scheduler：均匀分配，一个请求过来，就分配到后端instance上去
    * dynamic batch：打了batch后，再分配到后端instance上去
 
* stateful (predict result depends on previous sequence)
  * NLP
  * sequence batch。
    * direct, oldest
      * direct：
      * oldest：
   
* Ensemble
  * a pipeline of models
  * each model can have its own scheduler
 

Streaming Inference Requests
* based on correlation ID, the audio requests are sent to the appropriate batch slot in the sequence batcher

# Auxiliary Features of Triton

## model analyzer

* the model analyzer is a suite of tools that provides analysis and guidence on how to best optimize single or multiple models within triton, based on specific performance requirements
* it helps users make trade off decisions around throughput, latency, and gpu memory footprint when selecting the right model configurations
* two benchmarking functionalities avaliable:
  * performence analysis - measures throughput (infer/s) and latency under varying client loads
    * 一秒 100 请求，一秒 1000请求的 吞吐情况
  * memory analysis - measures a model's gpu memory footprint under different configurations
 

# Additional resources

* https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html
* https://docs.nvidia.com/deeplearning/triton-inference-server/index.html
* https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/triton/deployment_toolkit
* https://developer.nvidia.com/blog/maximizing-deep-learning-inference-performance-with-nvidia-model-analyzer/
* 

# In-house Inference Server vs Triton
> 和自研框架的比较

* open source with bsd license
  * community means virtual teamwork from lots of companies - minimize risk of architectural refactoring
  * core members from NV - best adaptation with nv gpu
  * contribution of customer backends from lots of companies
  * leverage new features, post issues
 
* support most of frameworks
* adopted by lots of companies - mature enough
* decoupled customer models / customer scenarios by writing customer backends
* with the support of multi-instance, grpc and cuda based memory optimization, gets better performance in most cases compared with in-house inference server
* metrics support


# step by step serve a model

* client
  * python / c++ client library
 
* server
  * do request
    * stadard http/grpc   OR  c api (directly integrate into client app)
    * dynamic batching (realtime, batch, stream)
    * per model scheduler queues
    * multiple gpu & cpu backends (tf, torch, onnx, trt)
  * response
    * stadard http/grpc   OR  c api (directly integrate into client app)
   
* model repository:
  * 根据 model 的 backends 进行 serving
 

1. prepare the model model reposity
2. configure the served model
3. launch triton server
4. configure an ensemble model
5. send requests to triton server


## prepare the model model reposity

```
# repo 目录结构
-- model-repo-path
   -- model-name
      -- config.pbtxt
      -- output-labels-file.txt
      -- version
         -- model-definition-file
      -- version
         -- model-definition-file

# repo 实例

-- model_repo
   -- densenet_onnx
      -- 1
         -- model.onnx
      -- config.pbtxt
      -- densenet_labels.txt
   -- inception_graphdef
      -- 1
         -- model.graphdef
      -- config.pbtxt
      -- inception_labels.txt
   -- resnet50_trt
      -- 1
         -- model.plan
   -- custome_model
      -- 1
         -- model.so
      -- config.pbtxt

```

* config 文件
* densenet_labels.txt 文件，分类的标签

模型配置介绍
* version directory：contains the model files (including resource files), directory name is version number, userd for version control
  * tensorrt: model.plan
  * onnx: model.onnx
  * torch-scripts: model.pt
  * tensorflow: model.graphdef or model.savedmodel
  * python: model.py
  * dali: model.dali
  * openvino: model.xml and model.bin
  * custom: model.so
 
* config file【optional】: defines configuration parameters for the model and server
* label file【optional】: for classification model, label file autoly covert predicted probs into label names

## configure the served model

* minimal model configuration
  * necessary parameters
    * platform / backend: to define which beckend to use。（一般2选一就可以）
    * max_batch_size: 主要用来限制模型推理的过程中不会超过 GPU显存 或者 CPU内存。（如果请求batchsize超过该配置，请求调用会报错）
    * input and output：输入和输出 tensor叫什么名字
   
  * tensorrt, tf-saved-model, and onnx models do not require config.pbtxt when --strict-model-config=false。启动tritonserver的时候把参数传入就可以了，如果不加，则必须提供 config.pbtxt
 
* platform / backend 指定：二者选一个指定就可以了
  * backend
    * tensorrt: tensorrt
    * onnx_rt: onnxruntime
    * tensorflow: tensorflow  (必须指定platform tensorflow_graphdef or tensorflow_savedmodel)
    * pytorch: pytorch
    * openvino: openvino
    * python: python
    * dali: dali
    * custome: ???

```

# demo1
backend: "tensorrt"
max_batch_size: 8  # 当maxbatchsize>0时，triton默认batchsize那一维是可变的，用户请求 [1, 3, 244, 244], [7, 3, 244, 244]都可以
input [
   {
      name: "input0"
      data_type: TYPE_FP32
      dims: [3, 244, 244]
   },

   {
      name: "input1"
      data_type: TYPE_FP32
      dims: [3, 244, 244]
   }
]

output [
   {
      name: "output0"
      data_type: TYPE_FP32
      dims: [16]
   }
]

# demo2
backend: "tensorrt"
max_batch_size: 0  # 当为0的时候，这个模型的输入shape就是[3, 244, 244],  如果想batch推理，那么 dims=[10, 3, 244, 244]。但是batchsize不可变
input [
   {
      name: "input0"
      data_type: TYPE_FP32
      dims: [3, 244, 244]
   },

   {
      name: "input1"
      data_type: TYPE_FP32
      dims: [3, 244, 244]
   }
]

output [
   {
      name: "output0"
      data_type: TYPE_FP32
      dims: [16]
   }
]

# demo 3
platform: "pytorch_libtorch"
max_batch_size: 8
input [
   {
      name: "input0"
      data_type: TYPE_FP32
      format: FORMAT_NCHW
      dims: [3, -1, -1]   # -1的用来标识 可变长度
   },

   {
      name: "input1"
      data_type: TYPE_FP32
      dims: [3, -1, -1]
   }
]

output [
   {
      name: "output0"
      data_type: TYPE_FP32
      dims: [16]
      reshape {shape:[1, 16]}
      label_filename: "labels.txt"
   }
]

# demo 4
backend: "tensorrt"
max_batch_size: 0
input [
   {
      name: "input0"
      data_type: TYPE_FP32
      dims: [3, 244, 244]
      reshape {shape: [1, 3, 244, 244]} # 能加一个reshape操作
   },

   {
      name: "input1"
      data_type: TYPE_FP32
      dims: [3, 244, 244]
   }
]

output [
   {
      name: "output0"
      data_type: TYPE_FP32
      dims: [16]
      reshape {shape: [1, 16, 1, 1]}
   }
]
```

### version policy

* all: all versions of the model that are available in the model repository are availiable for inferencing
* latest: only the latest 'n' versions of the model in the repo are avaliable for inferencing. the latest versions of the model are the numerically greatest version numbers
* specific: only the specifically listed versions of the model are available for inferencing

```
# 可选的，也是写在 config.pbtxt中，和 backend、max_batch_size, input, ouput同级

version_policy: {all {}}
version_policy: {latest {num_versions: 2}}
version_policy: {specific {versions: 1, 2}}
```

### instance groups
对应 triton的一个特性：concurrent model inference, 同一个模型可以开多个instance，由多个instance进行推理

```
# 也是和 backend 同级
instance_group [   # define a group of model insrances running on the same device
   {
      count: 2  # number of instance on each devices
      kind: KIND_CPU   # what kind of device to use
   }
   {
      count: 1
      kind: KIND_GPU
      gpus: [0]  # which gpus to use
   }

   {
      count: 2:
      kind: KIND_GPU
      gpus: [1, 2]  # 每个指定的gpu都会跑2个insrance。如果不指定的话，triton会再每个gpu上跑2个instance
   }
]
```

可以观察 gpu 利用率 来调整 一个gpu开多少instance
```shell
watch -n 1 nvidia-smi
```

```
--model-control-mode explicit # triton不会主动加载模型。客户端给特定请求，才会加载
```

## launch triton server
> tritonserver 可以自己编译，也可以使用 docker image

`tritonserver --model-reposity=/path/to/model/repo`

## configure an ensemble model

## send requests to triton server

# 参考资料

* https://www.bilibili.com/video/BV1R3411g7VR/?p=3&spm_id_from=pageDriver
