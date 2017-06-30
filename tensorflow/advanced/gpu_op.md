# GPU OP

> GPU kernel 的实现分为两个部分：OpKernel， CUDA kernel及其launch code。

很多情况下， `OpKernel` 的实现 ， `CPU`与`GPU`是通用的，例如，检查输入，给输出分配空间。在这种情况下，建议使用以下方法实现：

1. 定义 `OpKernel`模板，模板参数有 <Device, tensortype>
2. 计算输出，`Compute` 方法调用 一个 模板 `functor` 结构体
3. 给 CPUdevice 用的 functor 定义在同一文件下，给 GPU 用的放在一个 `.cu.cc` 文件中，因为这个文件要用 `nvcc` 编译。



```c++
// example.h
#ifndef KERNEL_EXAMPLE_H_
#define KERNEL_EXAMPLE_H_

template <typename Device, typename T>
struct ExampleFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#endif KERNEL_EXAMPLE_H_
```

```c++
// example.cc
#define EIGEN_USE_THREADS
#include "example.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T>
struct ExampleFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, const T* in, T* out) {
    for (int i = 0; i < size; ++i) {
      out[i] = 2 * in[i];
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class ExampleOp : public OpKernel {
 public:
  explicit ExampleOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    ExampleFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(input_tensor.NumElements()),
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Example").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ExampleOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Example").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ExampleOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
```

```c++
//example_gpu.cu.cc
#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "example.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#define EIGEN_USE_GPU

// Define the CUDA kernel.
template <typename T>
__global__ void ExampleCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = 2 * ldg(in + i);
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct ExampleFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int size, const T* in, T* out) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int block_count = 1024;
    int thread_per_block = 20;
    ExampleCudaKernel<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
  }
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;
template struct ExampleFunctor<GPUDevice, float>;
template struct ExampleFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA
```

