# tensorflow源码阅读



## Tensor

[地址](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.h)

* 保存了引用计数
* 值是存在 `TensorBuffer` 对象中的

## Op

[地址](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op.cc)

* `class  OpRegistryInterface` 查找 已注册的 `OpDef`
* `class OpRegistry : public OpRegistryInterface` 

```c++
class OpRegistry : public OpRegistryInterface {
 public:
  typedef std::function<Status(OpRegistrationData*)> OpRegistrationDataFactory;

  OpRegistry();
  ~OpRegistry() override;
  
  // 将 op_data_factory 添加到 deferred_ 成员中
  void Register(const OpRegistrationDataFactory& op_data_factory);

  Status LookUp(const string& op_type_name,
                const OpRegistrationData** op_reg_data) const override;

  // Fills *ops with all registered OpDefs (except those with names
  // starting with '_' if include_internal == false).
  // 看起来，这个是用来导出在 registry_ 成员中的 op 的
  void Export(bool include_internal, OpList* ops) const;

  // Returns ASCII-format OpList for all registered OpDefs (except
  // those with names starting with '_' if include_internal == false).
  string DebugString(bool include_internal) const;

  // A singleton available at startup. 单件模式，返回 OpRegistry 对象
  static OpRegistry* Global();

  // Get all registered ops.
  void GetRegisteredOps(std::vector<OpDef>* op_defs);

  // Watcher, a function object.
  // The watcher, if set by SetWatcher(), is called every time an op is
  // registered via the Register function. The watcher is passed the Status
  // obtained from building and adding the OpDef to the registry, and the OpDef
  // itself if it was successfully built. A watcher returns a Status which is in
  // turn returned as the final registration status.
  typedef std::function<Status(const Status&, const OpDef&)> Watcher;

  // An OpRegistry object has only one watcher. This interface is not thread
  // safe, as different clients are free to set the watcher any time.
  // Clients are expected to atomically perform the following sequence of
  // operations :
  // SetWatcher(a_watcher);
  // Register some ops;
  // op_registry->ProcessRegistrations();
  // SetWatcher(nullptr);
  // Returns a non-OK status if a non-null watcher is over-written by another
  // non-null watcher.
  Status SetWatcher(const Watcher& watcher);

  // Process the current list of deferred registrations. Note that calls to
  // Export, LookUp and DebugString would also implicitly process the deferred
  // registrations. Returns the status of the first failed op registration or
  // Status::OK() otherwise.
  Status ProcessRegistrations() const;

  // Defer the registrations until a later call to a function that processes
  // deferred registrations are made. Normally, registrations that happen after
  // calls to Export, LookUp, ProcessRegistrations and DebugString are processed
  // immediately. Call this to defer future registrations.
  void DeferRegistrations();

  // Clear the registrations that have been deferred.
  void ClearDeferredRegistrations();

 private:
  // Ensures that all the functions in deferred_ get called, their OpDef's
  // registered, and returns with deferred_ empty.  Returns true the first
  // time it is called. Prints a fatal log if any op registration fails.
  bool MustCallDeferred() const EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Calls the functions in deferred_ and registers their OpDef's
  // It returns the Status of the first failed op registration or Status::OK()
  // otherwise.
  Status CallDeferred() const EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Add 'def' to the registry with additional data 'data'. On failure, or if
  // there is already an OpDef with that name registered, returns a non-okay
  // status.
  Status RegisterAlreadyLocked(const OpRegistrationDataFactory& op_data_factory)
      const EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutable mutex mu_;
  // Functions in deferred_ may only be called with mu_ held.
  mutable std::vector<OpRegistrationDataFactory> deferred_ GUARDED_BY(mu_);
  // Values are owned. 注册的 op 都存在这里
  mutable std::unordered_map<string, const OpRegistrationData*> registry_
      GUARDED_BY(mu_);
  mutable bool initialized_ GUARDED_BY(mu_);

  // Registry watcher.
  mutable Watcher watcher_ GUARDED_BY(mu_);
};

// An adapter to allow an OpList to be used as an OpRegistryInterface.
//
// Note that shape inference functions are not passed in to OpListOpRegistry, so
// it will return an unusable shape inference function for every op it supports;
// therefore, it should only be used in contexts where this is okay.
class OpListOpRegistry : public OpRegistryInterface {
 public:
  // Does not take ownership of op_list, *op_list must outlive *this.
  OpListOpRegistry(const OpList* op_list);
  ~OpListOpRegistry() override;
  Status LookUp(const string& op_type_name,
                const OpRegistrationData** op_reg_data) const override;

 private:
  // Values are owned.
  std::unordered_map<string, const OpRegistrationData*> index_;
};

// Support for defining the OpDef (specifying the semantics of the Op and how
// it should be created) and registering it in the OpRegistry::Global()
// registry.  Usage:
// REGISTER_OP("name") 创建一个 OpDefBuilder 对象并返回
// REGISTER_OP("my_op_name")
//     .Attr("<name>:<type>")
//     .Attr("<name>:<type>=<default>")
//     .Input("<name>:<type-expr>")
//     .Input("<name>:Ref(<type-expr>)")
//     .Output("<name>:<type-expr>")
//     .Doc(R"(
// <1-line summary>
// <rest of the description (potentially many lines)>
// <name-of-attr-input-or-output>: <description of name>
// <name-of-attr-input-or-output>: <description of name;
//   if long, indent the description on subsequent lines>
// )");
//
// Note: .Doc() should be last.
// For details, see the OpDefBuilder class in op_def_builder.h.
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def_builder.h
```

[**OpDef**](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto)



**OpRegistrationData** [地址](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def_builder.h)

```c++
struct OpRegistrationData {
 public:
  OpRegistrationData() {}
  OpRegistrationData(const OpDef& def) : op_def(def) {}

  OpDef op_def;
  OpShapeInferenceFn shape_inference_fn;
};
```



**OpDefBuilder** [地址](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def_builder.h)

Constructs an OpDef with just the name field set

```c++
class OpDefBuilder {
 public:
  // Constructs an OpDef with just the name field set.
  explicit OpDefBuilder(StringPiece op_name);

  OpDefBuilder& Attr(StringPiece spec);


  OpDefBuilder& Input(StringPiece spec);
  OpDefBuilder& Output(StringPiece spec);

  OpDefBuilder& SetIsCommutative();
  OpDefBuilder& SetIsAggregate();
  OpDefBuilder& SetIsStateful();
  OpDefBuilder& SetAllowsUninitializedInput();


  OpDefBuilder& Deprecated(int version, StringPiece explanation);

#ifndef TF_LEAN_BINARY
  OpDefBuilder& Doc(StringPiece text);
#else
  OpDefBuilder& Doc(StringPiece text) { return *this; }
#endif

  OpDefBuilder& SetShapeFn(Status (*fn)(shape_inference::InferenceContext*));
  Status Finalize(OpRegistrationData* op_reg_data) const;

 private:
  OpDef* op_def() { return &op_reg_data_.op_def; }

  OpRegistrationData op_reg_data_;
  std::vector<string> attrs_;
  std::vector<string> inputs_;
  std::vector<string> outputs_;
  string doc_;
  std::vector<string> errors_;
};
```



## Graph

* Graph 由 NodeDef 构成

```c++
syntax = "proto3";

package tensorflow;
option cc_enable_arenas = true;
option java_outer_classname = "GraphProtos";
option java_multiple_files = true;
option java_package = "org.tensorflow.framework";

import "tensorflow/core/framework/node_def.proto";
import "tensorflow/core/framework/function.proto";
import "tensorflow/core/framework/versions.proto";

// Represents the graph of operations
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



## Variable

```c++
package tensorflow;
option cc_enable_arenas = true;
option java_outer_classname = "VariableProtos";
option java_multiple_files = true;
option java_package = "org.tensorflow.framework";

// Protocol buffer representing a Variable.
message VariableDef {
  // Name of the variable tensor.
  string variable_name = 1;

  // Name of the initializer op.
  string initializer_name = 2;

  // Name of the snapshot tensor.
  string snapshot_name = 3;

  // Support for saving variables as slices of a larger variable.
  SaveSliceInfoDef save_slice_info_def = 4;

  // Whether to represent this as a ResourceVariable.
  bool is_resource = 5;
}

message SaveSliceInfoDef {
  // Name of the full variable of which this is a slice.
  string full_name = 1;
  // Shape of the full variable.
  repeated int64 full_shape = 2;
  // Offset of this variable into the full variable.
  repeated int64 var_offset = 3;
  // Shape of this variable.
  repeated int64 var_shape = 4;
}
```



## NodeDef

```c++
syntax = "proto3";

package tensorflow;
option cc_enable_arenas = true;
option java_outer_classname = "NodeProto";
option java_multiple_files = true;
option java_package = "org.tensorflow.framework";

import "tensorflow/core/framework/attr_value.proto";

message NodeDef {
  // The name given to this operator. Used for naming inputs,
  // logging, visualization, etc.  Unique within a single GraphDef.
  // Must match the regexp "[A-Za-z0-9.][A-Za-z0-9_./]*".
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
  //              | ( ("gpu" | "cpu") ":" ([1-9][0-9]* | "*") )
  //
  // Valid values for this string include:
  // * "/job:worker/replica:0/task:1/gpu:3"  (full specification)
  // * "/job:worker/gpu:3"                   (partial specification)
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
};
```



## FunctionDefHelper

> FunctionDefHelper::Create is a convenient helper to construct a FunctionDef proto.
>
> FunctionDef --->  NodeDef 的集合

[FunctionDef](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/function.proto)



[FDH地址](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/function.h)

```c++
// FunctionDefHelper::Create is a convenient helper to construct a
// FunctionDef proto.
// E.g.,
//   FunctionDef my_func = FunctionDefHelper::Create(
//     "my_func_name",
//     {"x:T", "y:T" /* one string per argument */},
//     {"z:T" /* one string per return value */},
//     {"T: {float, double}" /* one string per attribute  */},
//     {
//        {{"o"}, "Mul", {"x", "y"}, {{"T", "$T"}}}
//        /* one entry per function node */
//     },
//     /* Mapping between function returns and function node outputs. */
//     {{"z", "o:z"}});
//
```

[FDH::Create](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/function.cc#L1082)

```c++
FunctionDef FunctionDefHelper::Create(
    const string& function_name, gtl::ArraySlice<string> in_def,
    gtl::ArraySlice<string> out_def, gtl::ArraySlice<string> attr_def,
    gtl::ArraySlice<Node> node_def,
    gtl::ArraySlice<std::pair<string, string>> ret_def) {
  FunctionDef fdef;

  // Signature, 建立了一个 OpDef
  OpDefBuilder b(function_name);
  for (const auto& i : in_def) b.Input(i);
  for (const auto& o : out_def) b.Output(o);
  for (const auto& a : attr_def) b.Attr(a);

  OpRegistrationData op_reg_data;
  TF_CHECK_OK(b.Finalize(&op_reg_data));
  fdef.mutable_signature()->Swap(&op_reg_data.op_def);

  // Function body
  for (const auto& n : node_def) {
    *(fdef.add_node_def()) = n.ToNodeDef();
  }

  // Returns
  for (const auto& r : ret_def) {
    fdef.mutable_ret()->insert({r.first, r.second});
  }
  return fdef;
}

/* static */
FunctionDef FunctionDefHelper::Define(const string& name,
                                      gtl::ArraySlice<string> arg_def,
                                      gtl::ArraySlice<string> ret_def,
                                      gtl::ArraySlice<string> attr_def,
                                      gtl::ArraySlice<Node> node_def) {
  FunctionDef fdef;
  OpDefBuilder b(name);
  for (const auto& a : arg_def) b.Input(a);
  for (const auto& r : ret_def) b.Output(r);
  for (const auto& a : attr_def) b.Attr(a);

  OpRegistrationData op_reg_data;
  TF_CHECK_OK(b.Finalize(&op_reg_data));
  fdef.mutable_signature()->Swap(&op_reg_data.op_def);

  // Mapping from legacy output names to NodeDef outputs.
  std::unordered_map<string, string> ret_index;
  for (const auto& a : fdef.signature().input_arg()) {
    ret_index[a.name()] = a.name();
  }

  // For looking up OpDefs
  auto* op_def_registry = OpRegistry::Global();

  // Function body
  for (const auto& src : node_def) {
    NodeDef* n = fdef.add_node_def();
    n->set_op(src.op);
    n->set_name(src.ret[0]);
    for (const auto& a : src.attr) {
      n->mutable_attr()->insert({a.first, a.second.proto});
    }
    for (const string& a : src.arg) {
      const auto iter = ret_index.find(a);
      CHECK(iter != ret_index.end()) << "Node input '" << a << "' in '"
                                     << src.ret[0] << "' of " << name;
      n->add_input(iter->second);
    }
    for (const string& d : src.dep) {
      n->add_input(strings::StrCat("^", d));
    }

    // Add the outputs of this node to ret_index.
    const OpDef* op_def = nullptr;
    TF_CHECK_OK(op_def_registry->LookUpOpDef(n->op(), &op_def)) << n->op();
    CHECK(op_def != nullptr) << n->op();
    NameRangeMap output_names;
    TF_CHECK_OK(NameRangesForNode(*n, *op_def, nullptr, &output_names));
    for (const auto& o : output_names) {
      CHECK_LE(o.second.second, src.ret.size())
          << "Missing ret for output '" << o.first << "' in '" << src.ret[0]
          << "' of " << name;
      for (int i = o.second.first; i < o.second.second; ++i) {
        ret_index[src.ret[i]] =
            strings::StrCat(src.ret[0], ":", o.first, ":", i - o.second.first);
      }
    }
  }

  // Returns
  for (const auto& r : fdef.signature().output_arg()) {
    const auto iter = ret_index.find(r.name());
    CHECK(iter != ret_index.end()) << "Return '" << r.name() << "' in " << name;
    fdef.mutable_ret()->insert({r.name(), iter->second});
  }
  return fdef;
}
```



## REGISTER_OP_GRADIENT

[地址](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/function.h#L488)

```c++
#define REGISTER_OP_GRADIENT(name, fn) \
  REGISTER_OP_GRADIENT_UNIQ_HELPER(__COUNTER__, name, fn)

#define REGISTER_OP_NO_GRADIENT(name) \
  REGISTER_OP_GRADIENT_UNIQ_HELPER(__COUNTER__, name, nullptr)

#define REGISTER_OP_GRADIENT_UNIQ_HELPER(ctr, name, fn) \
  REGISTER_OP_GRADIENT_UNIQ(ctr, name, fn)

#define REGISTER_OP_GRADIENT_UNIQ(ctr, name, fn)                 \
  static bool unused_grad_##ctr = SHOULD_REGISTER_OP_GRADIENT && \
                                  ::tensorflow::gradient::RegisterOp(name, fn)

namespace gradient {
// Register a gradient creator for the "op".
typedef std::function<Status(const AttrSlice& attrs, FunctionDef*)> Creator;
bool RegisterOp(const string& op, Creator func);

// Returns OK the gradient creator for the "op" is found (may be
// nullptr if REGISTER_OP_NO_GRADIENT is used.
Status GetOpGradientCreator(const string& op, Creator* creator);
};
```

```c++
namespace gradient {

// 传说中存 GradOp 的地方
typedef std::unordered_map<string, Creator> OpGradFactory;

OpGradFactory* GetOpGradFactory() {
  static OpGradFactory* factory = new OpGradFactory;
  return factory;
}

bool RegisterOp(const string& op, Creator func) {
  CHECK(GetOpGradFactory()->insert({op, func}).second)
      << "Duplicated gradient for " << op;
  return true;
}

Status GetOpGradientCreator(const string& op, Creator* creator) {
  auto fac = GetOpGradFactory();
  auto iter = fac->find(op);
  if (iter == fac->end()) {
    return errors::NotFound("No gradient defined for op: ", op);
  }
  *creator = iter->second;
  return Status::OK();
}

}  // end namespace gradient
```



## 重要类型一览

**Op相关**

* [OpRegistry](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op.h) : 注册 `op` 保存注册的 `op`, 注册的是 `OpRegistrationData`
* [OpDef](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto) : `proto` 
*  [OpRegistrationData](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def_builder.h) : 保存着 OpDef 
* [OpDefBuilder](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def_builder.h) : Constructs an OpDef with just the name field set



**Node相关**

* [NodeDef](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/node_def.proto) : 这个和 `OpDef` 是什么关系
* [Node]()
* [NodeDefBuilder](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/node_def_builder.h) : This is a helper for creating a NodeDef



## Op 与 Kernel 绑定

一个 `OP` 可以绑定多个 `Kernel` 

* [REGISTER_KERNLE_BUIDER](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_kernel.h#L1212)


* [KernelRegistry]()
* ​

```c++
void* GlobalKernelRegistry() {
  static KernelRegistry* global_kernel_registry = new KernelRegistry;
  return global_kernel_registry;
}
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_kernel.cc#L830
```



## executor

[地址](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc)

