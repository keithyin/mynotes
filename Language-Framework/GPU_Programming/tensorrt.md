# tensorrt整体介绍

tensorrt 做了啥：
1. 构建期
   1. 模型解析、建立 （加载onnx等其它格式的模型，使用原生API搭建模型）
   2.  计算图优化 （横向层融合Conv，纵向层融合Conv+add+Relu）。。。
   3.  节点消除    去除无用层，节点变换（Pad，Slice，Concat，Shuffle）
   4.  多精度支持  FP32，FP16，INT8，TF32（可能需要插入reformat节点）
   5.  优选Kernel、format 硬件相关优化
   6.  导入plugin    实现自定义操作（tensorrt无法直接支持的操作）
   7.  显存优化       显存池复用（会维护一个显存池，需要使用的内存会从显存池中读取）
9. 运行期
   1. 运行时环境       对象生命期管理，内存显存管理，异常处理
   2. 序列化，反序列化   推理引擎保存为文件或从文件中加载



Tensorrt的表现：
1. 不同模型的加速效果不同
2. 选用高效算子提升运算效率
3. 算子融合减少访存数据，提高访问效率
4. 使用低精度数据类型，节约时间空间


tensorrt代码基本流程：
1. 构建期
   1. 前期准备 （Logger，Builder，Config，Profile）
   2. 创建Network （计算图内容）
   3. 生成序列化网络
  
2. 运行期
   1. 建立engine 和 context
   2. Buffer相关准备 （申请 + 拷贝）
   3. 执行推理 （Execute）
   4. 善后工作


已经训练好的网络如何转成Tensorrt
1. 使用框架自带的trt接口（tf-trt，Torch-TensorRT）
2. 使用parser（Tf/Torch/... -> ONNX -> TensorRT）【推荐用法】
3. 使用TensorRT原生API搭建网络

问题：
1. 怎样从头开始写一个网络
2. 哪些代码是API搭建特有的，哪些是workflow通用的
3. 怎样让一个network跑起来
4. 用于推理的输入输出显存怎么准备
5. 构建引擎需要时间，怎么构建一次，反复使用
6. tensorrt的开发环境

tensorrt基本流程（详细）：
1. 构建阶段
   1. 建立Logger
   2. 建立Builder（网络元数据）和BuilderConfig（网络元数据的配置）
   3. 创建Network（计算图内容）
   4. 生成SerializedNetwork（网络的Trt内部表示）
  
2. 运行阶段
   1. 建立engine（可执行代码）
   2. 创建context（GPU进程，运行资源）
   3. Buffer准备（Host端+Device端）
   4. Buffer拷贝（Host to Device）
   5. 执行推理 （Execute）
   6. Buffer拷贝 （Device to Host）
   7. 善后工作

Logger日志记录器
1. `logger = trt.Logger(trt.Logger.VERBOSE)`
   1. VERBOSE
   2. INFO
   3. 上面两个用的比较多
   4. WARNING
   5. ERROR
   6. INTERNAL_ERRROR
   7. 多个Builder可以共享一个Logger

Builder 引擎构建器
1. `builder = trt.Builder(logger)`
2. 常用API
   1. `builder.create_network()` 创建trt网络对象
   2. `builder.create_optimization_profile()` 创建用于DynamicShape的输入配置器
   3. 【弃用】`builder.max_batch_size = 256`    指定最大的batchsize（Static Shape模式下使用）
   4. 【弃用】`builder.max_workspace_size = 1 << 30` 指定最大可用显存（Byte）
   5. 【弃用】`builder.fp16_mode = True/False`   开启/关闭 fp16模式
   6. 【弃用】`builder.int8_mode = True/False`   开启/关闭 int8模式
   7. 【弃用】`builder.strict_type_constraints = True/False`  开启/关闭 强制精度模式（通常不用）
   8. 【弃用】`builder.refittable = True/False`   开启/关闭 refit模式
   9. Dynamic Shape模式 **必须** 改用 builderConfig 来进行这些配置。上面弃用的也会转到 builderConfig中进行配置
  
BuilderConfig网络属性选项
1. `config = builder.create_builder_config()`
2. 常用成员
   1. `config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)`
   2. `config.flag = ` 设置标志位开关，如启闭 FP16/INT8, Refit模型，手工数据类型限制 等
      1. `config.flags = 1 << int(trt.BuilderFlag.FP16)`
   4. `config.int8_calibrator = `  指定 INT8-PTQ的 校正器
   5. `config.add_optimization_profile(..)`  添加用于DynamicShape的输入配置器
   6. `config.set_static_source/set_timing_cache/set_preview_feature`


Network具体构造 （用tensorrt搭建网络时才会涉及）
1. `network = builder.create_network()`
2. 常用参数
   1. `1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)` 使用 Explicit Batch 模式
  
3. 常用方法
   1. `network.add_input("inp", trt.float32, (3, 4, 5))` 标记网络输入张量
   2. `conv_layer = network.add_convolution_nd(xxx)`    添加各种网络层
   3. `network.mark_output(conv_layer.get_output(0))`   标记网络输出张量
  
4. 常用获取信息的成员
   1. `network.name, network.num_layers, network.num_inputs, network.num_outputs`
   2. `network.has_implicit_batch_dimension, network.has_explicit_precision`
  

Explicit Batch 模式 VS Implicit Batch模式
1. Explicit Batch为主流Network构建方法。Implicit Batch （builder.create_network(0)）仅用作后向兼容
2. 所有张量显式包含Batch维度，比ImplicitBatch模式多一维 （**多一个维度，那个维度上需要有具体值吗** 还是？就可以）
3. 需要使用 `builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))`
4. Explicit Batch能做，Implicit Batch不能做的事情
   1. BatchNormalization
   2. Reshape/Transpose/Reduce over Batch dimension
   3. Dynamic Shape模式
   4. Loop结构
   5. 一些Layer高级用法（ShuffleLayer.set_input）
  
5. 从Onnx导入的模型也默认使用 ExplicitBatch模式

DynamicShape模式
1. 适用于**张量形状在推理时才决定** 的网络
2. 除了Batch维，其它维度也可以推理时才决定
3. 需要Explicit Batch模式
4. 需要OptimizationProfile帮助网络优化
5. 需用 context.set_input_shape 绑定实际输入数据形状

Profile指定输入张量大小范围（Dynamic Shape）
1. `profile = builder.create_optimization_profile()`
2.常用方法
   1. `profile.set_shape(tensor_name, min_shape, common_shape, max_shape)` 给定输入张量 最小、最常见、最大尺寸
      1. `min_shape=(1, 1, 1), common_shape=(3, 4, 5), max_shape=(5, 6, 7)`
   3. `config.add_optimization_profile(profile)`  将设置的 profile 传递给 config 以创建网络
   4. `context.set_binding_shape(0, (1, 1, 1))`: 这样就可以用这个形状进行推理了


Layer 和 Tensor
> 使用trt搭网络的时候才会用到


FP16模式
1.  `config.flags = 1 << int(trt.BuilderFlag.FP16)`
2.  建立engine的时间比 FP32要长（更多kernel选择，需要插入reformat节点）
3.  Timeline中出现 cnhwToNchw 等kernel调用
4.  部分层可能精度下降导致较大误差
   1. 找到较大误差的层（用polygraphy等工具）
   2. 强制该层使用FP32计算
      1. `config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)`
      2. `layer.precision = trt.float32`
     
INT8模式-PTQ
1. 需要有校准集


INT8模式-QAT
1. 。。。

tensorrt运行期
1. 生成trt内部表示
   1. `serialized_network = builder.build_serialized_network(network, config)`
  
2. 生成 engine
   1. `engine = trt.Runtime(logger).deserialized_cuda_engine(serialized_network)`
      1. 常用成员：`engine.num_io_tensors   (engine绑定的输入输出张量总数，N+M), engine.num_layers`
      2. 常用方法:
         1. `engine.get_tensor_name(i)`
         2. `engine.get_tensor_dtype(t_name)`  
         3. `engine.get_tensor_shape(t_name)`  DynamicShape模式下结果可能含-1
         4. `engine.get_tensor_mode(t_name)` 输入张量还是输出张量
   3. `l_tensor_name = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]`
  
3. 创建 context （相当于CPU进程，负责资源管理）
   1. `context = engine.create_execution_context()`
   2. 常用方法
      1. `context.set_input_shape(t_name, shape_of_inp_tensor)`
      2. `context.get_tensor_shape(t_name)`
      3. `context.set_tensor_address(t_name, address)`
      4. `context.execute_v3(stream)`  explict batch模式的异步执行
  
4. 绑定输入输出 (Dynamic Shape模式 必须)
   1. `context.set_input_shape(l_tensor_name[0], [3, 4, 5])`  trt8.5开始 binding系列api全部 deprecated，换成tensor系列api
  
5. 准备 `buffer`
   1. `input_host = np.ascontiguousarray(input_data.reshape(-1))`
   2. `output_host = np.empty(context.get_tensor_shape(l_tensor_name[1]), trt.ntype(engine.get_tensor_dtype(l_tensor_name[1])))`
   3. `input_device = cudart.cudaMalloc(inputHost.nbytes)[1]`
   4. `output_device = cudart.cudaMalloc(outputHost.nbytes)[1]`
   5. `context.set_tensor_address(l_tensor_name[0], input_device)`   用到的GPU指针提前在这里设置，不再传入 execute_v3 函数
   6. `context.set_tensor_address(l_tensor_name[1], output_device)`
  
6. 执行计算
   1. `cudart.cudaMemcpy(input_device, input_host.ctypes.data, input_host.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)`
   2. `context.execute_async_v3(0)`
   3. `cudart.cudaMemcpy(output_host.ctypes.data, output_device, output_host.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)`
  
7. 释放显存
   1. `cudart.cudaFree(input_device)`
   2. `cudart.cudaFree(outptu_device)`


CUDA异构计算
1. 同事准备CPU端内存和GPU端显存
2. 计算前将 数据从内存拷贝到 显存
3. 计算过程中的 输入输出数据均在 GPU端读写
4. 计算完成后 结果从 显存 拷贝到 内存

engine构建一次，反复使用进行计算
1. 序列与反序列化
   1. 将serialized network保存为文件，下次跳过构建直接使用
   2. 注意环境统一（硬件环境+CUDA+CUDNN+tensorrt环境）
      1. engine包含硬件相关优化，不能跨硬件平台使用
      2. 不同tensorrt生成的engine不能相互兼容
      3. 同平台同环境多次生成的 engine 可能不同
     
   3. tensorrt runtime 与 engine 版本不同时的报错信息
      1. engine plan file is not compatible ...
      2. serialization error in deserialize ...
     
   4. 

使用ONNX
```python
torch.onnx.export(net,
   t.randn(1, 1, height, width, device="cuda"),
   "./model.onnx",
   example_outputs = [t.randn(1, 10, device="cuda"), t.randn(1, device="cuda")],
   input_names = ["x"],
   output_names = ["y", "z"],
   do_constant_folding = True,
   verbose = True,
   keep_initializers_as_inputs = True,
   opset_version = 12,
   dynamic_axes = {"x"{0: "nBatchSize"}, "z": {0: "nBatchSize"}}
)


with open(onnx_file, "rb") as model:
   if not parser.parse(model.read()):
      print("failed parsing onnx file")
      for err in range(parser.num_errors):
         print(parser.get_error(err))

context = engine.create_execution_context()

```

trtexec, onnx-graphsurgeon, plugin都是使用parser使用时的必备知识

tensorrt开发环境
1. 推荐使用 nvidia-optimized docker
2. python库
   1. nvidia-pyindex, cuda-python(pytyhon>=3.7), pycuda, onnx, onnx-surgeon, onnxruntime-gpu, opencv-python, polygraphy
  
3. 推荐使用
   1. 最新的tensorrt8，更多的图优化，优化过程和推理过程显存使用量更少
   2. buidlerConfig api, 功能覆盖旧版本的 builder API, 旧版 builder api将被废弃
   3. explicit batch + dynamic shape模式。onnx格式默认模式，灵活性和应用场景更广，使模型适应性更好
   4. cuda-python库，完整的cuda api支持，修复pycuda库可能存在的问题（如遇其它框架交互使用时的同步操作等）
  

# 开发辅助工具

1. trtexec （tensorrt命令行工具，主要的e2e性能测试工具）
2. netron（网络可视化）
3. onnx-graphsurgeon （onnx计算图编辑）
4. polygraphy （结果验证与定位，图优化）
5. nsight systems （性能分析）

希望解决的问题：
1. 不想写脚本来跑tensorrt （使用命令行）
2. 怎么进行简单的推理性能测试 （测试延迟，吞吐量等）
3. 网络结构可视化
4. 计算图上有哪些节点阻碍tensorrt自动优化
5. 怎么处理tensorrt不支持的网络结构
6. 怎么检验tensorrt上计算结果正确性、精度？
7. 怎么找出计算错误，精度不足的层
8. 怎么进行简单的计算图优化
9. 怎么找出最耗时的层

# 参考资料

1. https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/cookbook
2. 

