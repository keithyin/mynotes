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




# 参考资料

1. https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/cookbook
2. 

