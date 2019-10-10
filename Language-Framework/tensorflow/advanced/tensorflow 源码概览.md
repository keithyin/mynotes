# tensorflow 源码概览

* kernel注册
* 通过计算图找对应注册的kernel执行计算
* 参数如何保存



## Kernel相关

> 由于同一个kernel, 可能需要运行在不同的平台上(CPU, GPU, TPU), 所以 一个接口可能对应多个kernel

* 接口定义及注册
* 实际kernel定义及注册
* `OpDef, KernelDef 啥区别???`



**接口定义**

```c++
#define REGISTER_OP(name) REGISTER_OP_UNIQ_HELPER(__COUNTER__, name)
#define REGISTER_OP_UNIQ_HELPER(ctr, name) REGISTER_OP_UNIQ(ctr, name)
#define REGISTER_OP_UNIQ(ctr, name)                                          \
  static ::tensorflow::register_op::OpDefBuilderReceiver register_op##ctr    \
      TF_ATTRIBUTE_UNUSED =                                                  \
          ::tensorflow::register_op::OpDefBuilderWrapper<SHOULD_REGISTER_OP( \
              name)>(name)
```

```c++
// Template specialization that turns all calls into no-ops.
template <>
class OpDefBuilderWrapper<false> {
 public:
  explicit constexpr OpDefBuilderWrapper(const char name[]) {}
  OpDefBuilderWrapper<false>& Attr(StringPiece spec) { return *this; }
  OpDefBuilderWrapper<false>& Input(StringPiece spec) { return *this; }
  OpDefBuilderWrapper<false>& Output(StringPiece spec) { return *this; }
  OpDefBuilderWrapper<false>& SetIsCommutative() { return *this; }
  OpDefBuilderWrapper<false>& SetIsAggregate() { return *this; }
  OpDefBuilderWrapper<false>& SetIsStateful() { return *this; }
  OpDefBuilderWrapper<false>& SetAllowsUninitializedInput() { return *this; }
  OpDefBuilderWrapper<false>& Deprecated(int, StringPiece) { return *this; }
  OpDefBuilderWrapper<false>& Doc(StringPiece text) { return *this; }
  OpDefBuilderWrapper<false>& SetShapeFn(
      Status (*fn)(shape_inference::InferenceContext*)) {
    return *this;
  }
};
```

```c++
struct OpDefBuilderReceiver {
  // To call OpRegistry::Global()->Register(...), used by the
  // REGISTER_OP macro below.
  // Note: These are implicitly converting constructors.
  OpDefBuilderReceiver(
      const OpDefBuilderWrapper<true>& wrapper);  // NOLINT(runtime/explicit)
  constexpr OpDefBuilderReceiver(const OpDefBuilderWrapper<false>&) {
  }  // NOLINT(runtime/explicit)
};
```

```c++
OpDefBuilderReceiver::OpDefBuilderReceiver(
    const OpDefBuilderWrapper<true>& wrapper) {
  OpRegistry::Global()->Register(
      [wrapper](OpRegistrationData* op_reg_data) -> Status {
        return wrapper.builder().Finalize(op_reg_data);
      });
}
OpRegistry* OpRegistry::Global() {
  static OpRegistry* global_op_registry = new OpRegistry;
  return global_op_registry;
}
```



**Kernel注册**

```c++
#define REGISTER_KERNEL_BUILDER(kernel_builder, ...) \
  REGISTER_KERNEL_BUILDER_UNIQ_HELPER(__COUNTER__, kernel_builder, __VA_ARGS__)

#define REGISTER_KERNEL_BUILDER_UNIQ_HELPER(ctr, kernel_builder, ...) \
  REGISTER_KERNEL_BUILDER_UNIQ(ctr, kernel_builder, __VA_ARGS__)

#define REGISTER_KERNEL_BUILDER_UNIQ(ctr, kernel_builder, ...)        \
  constexpr bool should_register_##ctr##__flag =                      \
      SHOULD_REGISTER_OP_KERNEL(#__VA_ARGS__);                        \
  static ::tensorflow::kernel_factory::OpKernelRegistrar              \
      registrar__body__##ctr##__object(                               \
          should_register_##ctr##__flag                               \
              ? ::tensorflow::register_kernel::kernel_builder.Build() \
              : nullptr,                                              \
          #__VA_ARGS__,                                               \
          [](::tensorflow::OpKernelConstruction* context)             \
              -> ::tensorflow::OpKernel* {                            \
            return new __VA_ARGS__(context);                          \
          });
```

```c++
class OpKernelRegistrar {
 public:
  // Registers the given kernel factory with TensorFlow. TF will call the
  // factory Create() method when it determines that a kernel matching the given
  // KernelDef is required.
  OpKernelRegistrar(const KernelDef* kernel_def, StringPiece kernel_class_name,
                    std::unique_ptr<OpKernelFactory> factory) {
    // Perform the check in the header to allow compile-time optimization
    // to a no-op, allowing the linker to remove the kernel symbols.
    if (kernel_def != nullptr) {
      InitInternal(kernel_def, kernel_class_name, std::move(factory));
    }
  }

  // Registers the given factory function with TensorFlow. This is equivalent
  // to registering a factory whose Create function invokes `create_fn`.
  OpKernelRegistrar(const KernelDef* kernel_def, StringPiece kernel_class_name,
                    OpKernel* (*create_fn)(OpKernelConstruction*)) {
    // Perform the check in the header to allow compile-time optimization
    // to a no-op, allowing the linker to remove the kernel symbols.
    if (kernel_def != nullptr) {
      InitInternal(kernel_def, kernel_class_name,
                   absl::make_unique<PtrOpKernelFactory>(create_fn));
    }
  }

 private:
  struct PtrOpKernelFactory : public OpKernelFactory {
    explicit PtrOpKernelFactory(OpKernel* (*create_func)(OpKernelConstruction*))
        : create_func_(create_func) {}

    OpKernel* Create(OpKernelConstruction* context) override;

    OpKernel* (*create_func_)(OpKernelConstruction*);
  };
  void InitInternal(const KernelDef* kernel_def, StringPiece kernel_class_name,
                    std::unique_ptr<OpKernelFactory> factory);
};
```

```c++
void OpKernelRegistrar::InitInternal(const KernelDef* kernel_def,
                                     StringPiece kernel_class_name,
                                     std::unique_ptr<OpKernelFactory> factory) {
  // See comments in register_kernel::Name in header for info on _no_register.
  if (kernel_def->op() != "_no_register") {
    const string key =
        Key(kernel_def->op(), DeviceType(kernel_def->device_type()),
            kernel_def->label());

    // To avoid calling LoadDynamicKernels DO NOT CALL GlobalKernelRegistryTyped
    // here.
    // InitInternal gets called by static initializers, so it ends up executing
    // before main. This causes LoadKernelLibraries function to get called
    // before some file libraries can initialize, which in turn crashes the
    // program flakily. Until we get rid of static initializers in kernel
    // registration mechanism, we have this workaround here.
    auto global_registry =
        reinterpret_cast<KernelRegistry*>(GlobalKernelRegistry());
    mutex_lock l(global_registry->mu);
    global_registry->registry.emplace(
        key,
        KernelRegistration(*kernel_def, kernel_class_name, std::move(factory)));
  }
  delete kernel_def;
}
```

* KernelDef

```protobuf
message KernelDef {
  // Must match the name of an Op.
  string op = 1;

  // Type of device this kernel runs on.
  string device_type = 2;

  message AttrConstraint {
    // Name of an attr from the Op.
    string name = 1;

    // A list of values that this kernel supports for this attr.
    // Like OpDef.AttrDef.allowed_values, except for kernels instead of Ops.
    AttrValue allowed_values = 2;
  }
  repeated AttrConstraint constraint = 3;

  // Names of the Op's input_/output_args that reside in host memory
  // instead of device memory.
  repeated string host_memory_arg = 4;

  // This allows experimental kernels to be registered for an op that
  // won't be used unless the user specifies a "_kernel" attr with
  // value matching this.
  string label = 5;

  // Prioritization of kernel amongst different devices. By default we assume
  // priority is 0. The higher the priority the better. By default (i.e. if
  // this is not set), we prefer GPU kernels over CPU.
  int32 priority = 6;
}
```

* KernelDefBuilder
  * 用来构建`KernelDef`对象的
  * `class Name : public KernelDefBuilder`
* OpKernelRegistrar
  * constructor `KernelDefBuilder, OpKernel(Kernel的实际实现)`
  * `InitInternal` 中往 `KernelRegistry` 中注册 `KernelOp`
* GlobalKernelRegistry: 返回`kernelRegistry对象` 的函数
* KernelRegistration : 封装被注册的 `OpKernel` 的 `kernelDef, kernel_class_name, factory(创建对象的工厂函数)`

```c++
struct KernelRegistration {
  KernelRegistration(const KernelDef& d, StringPiece c,
                     std::unique_ptr<kernel_factory::OpKernelFactory> f)
      : def(d), kernel_class_name(c), factory(std::move(f)) {}

  const KernelDef def;
  const string kernel_class_name;
  std::unique_ptr<kernel_factory::OpKernelFactory> factory;
};
```

* KernelRegistry

```c++
struct KernelRegistry {
  mutex mu;
  std::unordered_multimap<string, KernelRegistration> registry GUARDED_BY(mu);
};
```

* PtrOpKernelFactory



**REGISTER_KERNEL_BUILDER的流程为**

* `OpKernelRegistrar(KernelDefBuilder(), OpKernelName, factory)`
  * `OpKernelRegistrar::InitInternal`
    * 通过 `KernelDefBuilder` 构建的 `KernelDef` 构建 `key(op_name, device, label) `
    * 将  `factory` 注册到 `KernelRegistry` 中



## Session

[DirectSession](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.h)

* `GraphDef`

```protobuf
message GraphDef {
  repeated NodeDef node = 1;

  // Compatibility versions of the graph.  See core/public/version.h for version
  // history.  The GraphDef version is distinct from the TensorFlow version, and
  // each release of TensorFlow will support a range of GraphDef versions.
  VersionDef versions = 4;

  // Deprecated single version field; use versions above instead.  Since all
  // GraphDef changes before "versions" was introduced were forward
  // compatible, this field is entirely ignored.
  int32 version = 3 [deprecated = true];

  // EXPERIMENTAL. DO NOT USE OR DEPEND ON THIS YET.
  //
  // "library" provides user-defined functions.
  //
  // Naming:
  //   * library.function.name are in a flat namespace.
  //     NOTE: We may need to change it to be hierarchical to support
  //     different orgs. E.g.,
  //     { "/google/nn", { ... }},
  //     { "/google/vision", { ... }}
  //     { "/org_foo/module_bar", { ... }}
  //     map<string, FunctionDefLib> named_lib;
  //   * If node[i].op is the name of one function in "library",
  //     node[i] is deemed as a function call. Otherwise, node[i].op
  //     must be a primitive operation supported by the runtime.
  //
  //
  // Function call semantics:
  //
  //   * The callee may start execution as soon as some of its inputs
  //     are ready. The caller may want to use Tuple() mechanism to
  //     ensure all inputs are ready in the same time.
  //
  //   * The consumer of return values may start executing as soon as
  //     the return values the consumer depends on are ready.  The
  //     consumer may want to use Tuple() mechanism to ensure the
  //     consumer does not start until all return values of the callee
  //     function are ready.
  FunctionDefLibrary library = 2;
};
```

* `NodeDef`

```protobuf
message NodeDef {
  // The name given to this operator. Used for naming inputs,
  // logging, visualization, etc.  Unique within a single GraphDef.
  // Must match the regexp "[A-Za-z0-9.][A-Za-z0-9_>./]*".
  string name = 1;

  // The operation name.  There may be custom parameters in attrs.
  // Op names starting with an underscore are reserved for internal use.
  string op = 2;

  // Each input is "node:src_output" with "node" being a string name and
  // "src_output" indicating which output tensor to use from "node". If
  // "src_output" is 0 the ":0" suffix can be omitted.  Regular inputs
  // may optionally be followed by control inputs that have the format
  // "^node".
  repeated string input = 3;

  // A (possibly partial) specification for the device on which this
  // node should be placed.
  // The expected syntax for this string is as follows:
  //
  // DEVICE_SPEC ::= PARTIAL_SPEC
  //
  // PARTIAL_SPEC ::= ("/" CONSTRAINT) *
  // CONSTRAINT ::= ("job:" JOB_NAME)
  //              | ("replica:" [1-9][0-9]*)
  //              | ("task:" [1-9][0-9]*)
  //              | ("device:" [A-Za-z]* ":" ([1-9][0-9]* | "*") )
  //
  // Valid values for this string include:
  // * "/job:worker/replica:0/task:1/device:GPU:3"  (full specification)
  // * "/job:worker/device:GPU:3"                   (partial specification)
  // * ""                                    (no specification)
  //
  // If the constraints do not resolve to a single device (or if this
  // field is empty or not present), the runtime will attempt to
  // choose a device automatically.
  string device = 4;

  // Operation-specific graph-construction-time configuration.
  // Note that this should include all attrs defined in the
  // corresponding OpDef, including those with a value matching
  // the default -- this allows the default to change and makes
  // NodeDefs easier to interpret on their own.  However, if
  // an attr with a default is not specified in this list, the
  // default will be used.
  // The "names" (keys) must match the regexp "[a-z][a-z0-9_]+" (and
  // one of the names from the corresponding OpDef's attr field).
  // The values must have a type matching the corresponding OpDef
  // attr's type field.
  // TODO(josh11b): Add some examples here showing best practices.
  map<string, AttrValue> attr = 5;

  message ExperimentalDebugInfo {
    // Opaque string inserted into error messages created by the runtime.
    //
    // This is intended to store the list of names of the nodes from the
    // original graph that this node was derived. For example if this node, say
    // C, was result of a fusion of 2 nodes A and B, then 'original_node' would
    // be {A, B}. This information can be used to map errors originating at the
    // current node to some top level source code.
    repeated string original_node_names = 1;

    // This is intended to store the list of names of the functions from the
    // original graph that this node was derived. For example if this node, say
    // C, was result of a fusion of node A in function FA and node B in function
    // FB, then `original_funcs` would be {FA, FB}. If the node is in the top
    // level graph, the `original_func` is empty. This information, with the
    // `original_node_names` can be used to map errors originating at the
    // current ndoe to some top level source code.
    repeated string original_func_names = 2;
  };

  // This stores debug information associated with the node.
  ExperimentalDebugInfo experimental_debug_info = 6;
};
```



```python
import tensorflow as tf

a = tf.constant([1., 1.])
b = tf.constant([2., 2.])
c = a + b
with tf.Session() as sess:
    print(sess.graph_def)
```

以下是输出结果:

```json
node {
  name: "Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\200?\000\000\200?"
      }
    }
  }
}
node {
  name: "Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\000@\000\000\000@"
      }
    }
  }
}
node {
  name: "add"
  op: "Add"
  input: "Const"
  input: "Const_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
versions {
  producer: 38
}

```



* `GraphExecutionState` 通过 `GraphDef` 构建 `Graph`
  * 内部调用 `GraphConstructor::Construct`
* `GraphConstructor`: 通过 `GraphDef` 构建 `Graph`

