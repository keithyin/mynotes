# 数据结构

## data related

* `Tensor`

```c++
class CAFFE2_API Tensor {
public:
  Tensor(){};
  Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> tensor_impl)
      : impl_(std::move(tensor_impl)) {
    if (impl_.get() == nullptr) {
      throw std::runtime_error("TensorBaseImpl with nullptr not supported");
    }
  }

  Tensor(const Tensor&) = default;
  Tensor(Tensor&&) = default;

  int64_t dim() const {
    return impl_->dim();
  }
  int64_t storage_offset() const {
    return impl_->storage_offset();
  }

  TensorImpl * unsafeGetTensorImpl() const {
    return impl_.get();
  }
  TensorImpl * unsafeReleaseTensorImpl() {
    return impl_.release();
  }
  const c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>& getIntrusivePtr() const {
    return impl_;
  }

  bool defined() const {
    return impl_;
  }

  void reset() {
    impl_.reset();
  }
protected:
  c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> impl_;
};
```

* `TensorImpl`

```c++
struct CAFFE2_API TensorImpl : public c10::intrusive_ptr_target {
  TensorImpl() = delete;

  /**
   * Construct a 1-dim 0-size tensor with the given settings.
   * The provided allocator will be used to allocate data on
   * subsequent resize.
   */
  TensorImpl(TensorTypeId type_id, const caffe2::TypeMeta& data_type, Allocator *allocator, bool is_variable);

  /**
   * Construct a 1-dim 0-size tensor backed by the given storage.
   */
  TensorImpl(Storage&& storage, TensorTypeId type_id, bool is_variable);
public:
  at::Storage storage_; // TODO: Fix visibility on me
  
protected:
  // We could save a word or two by combining the SmallVector structs,
  // since their size is redundant, and if we need to overflow the buffer space
  // we could keep the two pointers together. However, that would require
  // implementing another struct from scratch, so only do this if we're desperate.
  at::SmallVector<int64_t,5> sizes_;
  at::SmallVector<int64_t,5> strides_;

  int64_t storage_offset_ = 0;
  // If sizes and strides are empty, the numel is 1!!  However, most of the
  // time, we will immediately set sizes to {0} and reset numel to 0.
  // (Can't do that in the default initializers, because there's no way to
  // spell "allocate a one-element array" for strides_).
  int64_t numel_ = 1;

  // INVARIANT: When storage is non-null, this type meta must
  // agree with the type meta in storage
  caffe2::TypeMeta data_type_;

  // You get to have eight byte-size fields here, before you
  // should pack this into a bitfield.
  TensorTypeId type_id_;
  bool is_contiguous_ = true;
  bool is_variable_ = false;
  bool is_wrapped_number_ = false;
  // we decide to keep reserved_ and it will
  // live in Tensor after the split
  // The logic is that if Extend() or ReserveSpace() were ever called,
  // then subsequent Resize()s will not free up Storage.
  bool reserved_ = false;

};
```



* `Storage`

```c++
struct C10_API Storage {
 public:
  Storage() {}
  Storage(c10::intrusive_ptr<StorageImpl> ptr) : storage_impl_(std::move(ptr)) {}
  Storage(
      caffe2::TypeMeta data_type,
      size_t size,
      Allocator* allocator,
      bool resizable = false)
      : storage_impl_(c10::make_intrusive<StorageImpl>(
            data_type,
            size,
            allocator,
            resizable)) {}

  Storage(
      caffe2::TypeMeta data_type,
      at::DataPtr data_ptr,
      size_t size,
      const std::function<void(void*)>& deleter,
      bool resizable = false)
      : storage_impl_(c10::make_intrusive<StorageImpl>(
            data_type,
            size,
            std::move(data_ptr),
            /* allocator */ nullptr,
            resizable)) {}

  Storage(at::DeviceType device_type)
      : storage_impl_(
            c10::make_intrusive<StorageImpl>(at::Device(device_type))) {}
  Storage(at::Device device)
      : storage_impl_(c10::make_intrusive<StorageImpl>(device)) {}
  Storage(at::Device device, caffe2::TypeMeta data_type)
      : storage_impl_(c10::make_intrusive<StorageImpl>(device, data_type)) {}

  Storage(
      caffe2::TypeMeta data_type,
      int64_t numel,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable)
      : storage_impl_(c10::make_intrusive<StorageImpl>(
            data_type,
            numel,
            std::move(data_ptr),
            allocator,
            resizable)) {}

  template <typename T>
  inline bool IsType() const {
    return storage_impl_->IsType<T>();
  }

  template <typename T>
  T* data() const { return storage_impl_->data<T>(); }

  template <typename T>
  T* unsafe_data() const { return storage_impl_->unsafe_data<T>(); }

  size_t elementSize() const {
    return storage_impl_->itemsize();
  }

  inline size_t itemsize() const {
    return storage_impl_->itemsize();
  }

  ptrdiff_t size() const {
    return storage_impl_->numel();
  }

  int64_t numel() const {
    return storage_impl_->numel();
  }

  // TODO: remove later
  void set_numel(int64_t numel) const {
    storage_impl_.get()->set_numel(numel);
  }

  bool resizable() const {
    return storage_impl_->resizable();
  }

  size_t capacity() const {
    return storage_impl_->capacity();
  }
  // get() use here is to get const-correctness

  void* data() const {
    return storage_impl_.get()->data();
  }

  const caffe2::TypeMeta& dtype() const {
    return storage_impl_->dtype();
  }

  at::DataPtr& data_ptr() {
    return storage_impl_->data_ptr();
  }

  const at::DataPtr& data_ptr() const {
    return storage_impl_->data_ptr();
  }
  operator bool() const {
    return storage_impl_;
  }

  size_t use_count() const {
    return storage_impl_.use_count();
  }

  std::move(data_ptr), data_type, capacity);
  }

 protected:
  c10::intrusive_ptr<StorageImpl> storage_impl_;
};
```

* `StorageImpl`

```c++
struct C10_API StorageImpl final : public c10::intrusive_ptr_target {
 public:
  StorageImpl(
      caffe2::TypeMeta data_type,
      int64_t numel,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable)
      : data_type_(data_type),
        data_ptr_(std::move(data_ptr)),
        numel_(numel),
        resizable_(resizable),
        allocator_(allocator) {
    if (numel > 0) {
      if (data_type_.id() == caffe2::TypeIdentifier::uninitialized()) {
        AT_ERROR(
            "Constructing a storage with meta of unknown type and non-zero numel");
      }
    }
  }

  StorageImpl(
      caffe2::TypeMeta data_type,
      int64_t numel,
      at::Allocator* allocator,
      bool resizable)
      : StorageImpl(
            data_type,
            numel,
            allocator->allocate(data_type.itemsize() * numel),
            allocator,
            resizable) {}

  explicit StorageImpl(at::Device device)
      : StorageImpl(device, caffe2::TypeMeta()) {}

  StorageImpl(at::Device device, caffe2::TypeMeta data_type)
      : StorageImpl(data_type, 0, at::DataPtr(nullptr, device), nullptr, true) {
  }

 

  void reset() {
    data_ptr_.clear();
    numel_ = 0;
  }

  void release_resources() override {
    data_ptr_.clear();
  }

  bool resizable() const {
    return resizable_;
  };

  at::DataPtr& data_ptr() {
    return data_ptr_;
  };

  const at::DataPtr& data_ptr() const {
    return data_ptr_;
  };

  // XXX: TERRIBLE! DONT USE UNLESS YOU HAVE TO! AND EVEN THEN DONT, JUST DONT!
  // Setting the data_type will require you to audit many other parts of the
  // struct again to make sure it's still valid.
  void set_dtype(const caffe2::TypeMeta& data_type) {
    int64_t capacity = numel_ * data_type_.itemsize();
    data_type_ = data_type;
    numel_ = capacity / data_type_.itemsize();
  }

  // TODO: Return const ptr eventually if possible
  void* data() {
    return data_ptr_.get();
  }

  void* data() const {
    return data_ptr_.get();
  }

  at::DeviceType device_type() const {
    return data_ptr_.device().type();
  }

  at::Allocator* allocator() {
    return allocator_;
  }

  const caffe2::TypeMeta& dtype() const {
    return data_type_;
  }

  const at::Allocator* allocator() const {
    return allocator_;
  };

  // You generally shouldn't use this method, but it is occasionally
  // useful if you want to override how a tensor will be reallocated,
  // after it was already allocated (and its initial allocator was
  // set)
  void set_allocator(at::Allocator* allocator) {
    allocator_ = allocator;
  }

 private:
  caffe2::TypeMeta data_type_;
  DataPtr data_ptr_;
  int64_t numel_;
  bool resizable_;
  Allocator* allocator_;
};
```



* `InputBuffer` :  用来累积 `grad_fn` 的输入梯度

```c++
// The InputBuffer class accumulates a list of Variables for use by a
// function. It implements logic to avoid modifying the passed
// values in-place (adding an input twice will accumulate the result).
// This behaviour is needed and used only in backward graphs.
struct InputBuffer {
  explicit InputBuffer(size_t size)
    : buffer(size) {}
  InputBuffer(const InputBuffer& other) = delete;
  InputBuffer(InputBuffer&& other) = default;
  InputBuffer& operator=(InputBuffer&& other) = default;

  // Accumulates the variable at a specified index.
  void add(size_t pos, Variable var);

  int device() const;

  Variable operator[](size_t pos) { return buffer[pos]; }

  // Returns the inputs as a list of variables. Destroys given InputBuffer.
  static std::vector<Variable> variables(InputBuffer&& g);

private:
  std::vector<Variable> buffer;
};
```



* `SavedVariable`： 一般是用来存在 反向传导的`Function` 中用的。
  * `unpack` 中进行 `version` 和 `data_ 有没释放掉的校验`  

```c++
/// A snapshot of a variable at a certain version. A `SavedVariable` stores
/// enough information to reconstruct a variable from a certain point in time.

class TORCH_API SavedVariable {
 public:
  SavedVariable() = default;
  SavedVariable(const Variable& variable, bool is_output);
  SavedVariable(SavedVariable&&) = default;
  SavedVariable& operator=(SavedVariable&&) = default;

  /// Reconstructs the saved variable. Pass `saved_for` as the gradient
  /// function if constructing the `SavedVariable` with it would have caused a
  /// circular reference.
  Variable unpack(std::shared_ptr<Function> saved_for = nullptr) const;

  void reset_data() {
    return data_.reset();
  }

  void reset_grad_function() {
    grad_fn_.reset();
  }

 private:
  at::Tensor data_;

  // The gradient function associated with this node. If has_grad_fn
  // is false, then this is a leaf node. Note that the grad_fn is not saved if
  // it would create a circular reference. In that case, the grad_fn must be
  // passed in to the unpack function when reconstructing the Variable.
  std::shared_ptr<Function> grad_fn_;
  std::weak_ptr<Function> grad_accumulator_;
  VariableVersion version_counter_;

  uint32_t saved_version_ = 0;
  uint32_t output_nr_ = 0;
  bool was_default_constructed_ = true;
  bool requires_grad_ = false;
  bool has_grad_fn_ = false;
};
```







## Function Related

> pytorch 反向求导引擎中，Function 作为节点
>
> * `GraphRoot` 是根节点：没有输入，只有输出（head gradient）。
> * `AccumulateGrad`: 是叶子节点，是 **sink** ，只有输入，没有输出。仅用来累积模型参数的梯度。



* `GraphRoot` 

```c++
/*
edge_list functions: 设置 next_edges 。
	- 调用 backward 的 variable 和 其 grad_fn 构成的 edges
variable_list inputs: head gradient
*/
struct TORCH_API GraphRoot : public Function {
  GraphRoot(edge_list functions, variable_list inputs)
      : Function(std::move(functions)),
        outputs(std::move(inputs)) {}

  variable_list apply(variable_list&& inputs) override {
    return outputs;
  }

  variable_list outputs;
};
```

* `AccumulateGrad`  : Function Graph 的叶子节点，用于累积 模型 variable 的梯度

```c++
struct AccumulateGrad : public Function {
  explicit AccumulateGrad(Variable variable_);

  variable_list apply(variable_list&& grads) override;

  Variable variable;
};
```



