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
基于推理的lifecycle来设计triton, 从推理请求开始 到 结束，会有哪些步骤，根据步骤设计系统
1. multi model frameworks supported -> this is subject to frameworks, should be decoupled by frameworks -> Backends
2. common functionalities
   1. common backend management
      1. backend load, etc.
     
   2. model management - load/unload, model update, query model status, etc.
   3. concurrent model execution - instance management
   4. request queue dispatch and scheduler
   5. infer lifecycle management
      1. inference request management
      2. inference response management
     
   6. GRPC related
      1. grpc server


# Auxiliary Features of Triton

# Additional resources

# In-house Inference Server vs Triton
