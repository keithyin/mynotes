# cudapython
> cuda 的 python binding

## 使用cudapython写kernel

1. 写 kernel 代码。（将c代码放到python string 里）
2. 使用 nvtrc 将kernel代码编译成 ptx (parallel thread execution, low-level virtual machine and instruction set architecture)
3. 构建一个 cuda context & 分配所需资源.
4. 执行编译的 c 代码，然后将结果从 gpu 中取出

```python
from cuda import cuda, nvrtc
import numpy as np

# 异常处理
def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


# 写cuda代码，放在""" """ 里
saxpy = """\
extern "C" __global__
void saxpy(float a, float *x, float *y, float *out, size_t n)
{
 size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
 if (tid < n) {
   out[tid] = a * x[tid] + y[tid];
 }
}
"""

# 使用 nvrtc 将 cuda 代码编译成 ptx
# 1. create program from string
# 2. compile the program
# 3. extract ptx from the program

# Create program
err, prog = nvrtc.nvrtcCreateProgram(str.encode(saxpy), b"saxpy.cu", 0, [], [])

# Compile program
opts = [b"--fmad=false", b"--gpu-architecture=compute_75"]
err, = nvrtc.nvrtcCompileProgram(prog, 2, opts)

# Get PTX from compilation
err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
ptx = b" " * ptxSize
err, = nvrtc.nvrtcGetPTX(prog, ptx)


# 在gpu上执行操作之前，必须要创建一个cuda context，cuda context类似于 设备的 主机进程！创建了 cuda context 后，可以继续来编译 kernel 了
# Initialize CUDA Driver API
err, = cuda.cuInit(0)

# Retrieve handle for device 0
err, cuDevice = cuda.cuDeviceGet(0)

# Create context
err, context = cuda.cuCtxCreate(0, cuDevice)


# 通过 gpu0 上的 cuda context，将 ptx 加载到 module（可以理解为动态链接库）上。加载到module上之后，就可以将对应的函数抽取出来
# Load PTX as module data and retrieve function
ptx = np.char.array(ptx)
# Note: Incompatible --gpu-architecture would be detected here
err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
ASSERT_DRV(err)
err, kernel = cuda.cuModuleGetFunction(module, b"saxpy")
ASSERT_DRV(err)


# 准备数据，然后准备执行 kernel
NUM_THREADS = 512  # Threads per block
NUM_BLOCKS = 32768  # Blocks per grid

a = np.array([2.0], dtype=np.float32)
n = np.array(NUM_THREADS * NUM_BLOCKS, dtype=np.uint32)
bufferSize = n * a.itemsize

hX = np.random.rand(n).astype(dtype=np.float32)
hY = np.random.rand(n).astype(dtype=np.float32)
hOut = np.zeros(n).astype(dtype=np.float32)

## 分配gpu内存
err, dXclass = cuda.cuMemAlloc(bufferSize)
err, dYclass = cuda.cuMemAlloc(bufferSize)
err, dOutclass = cuda.cuMemAlloc(bufferSize)

err, stream = cuda.cuStreamCreate(0)
## numpy 对象通过 hX.ctypes.data 来获取指针
err, = cuda.cuMemcpyHtoDAsync(
   dXclass, hX.ctypes.data, bufferSize, stream
)
err, = cuda.cuMemcpyHtoDAsync(
   dYclass, hY.ctypes.data, bufferSize, stream
)

## 构建kernel的调用参数
# The following code example is not intuitive 
# Subject to change in a future release
dX = np.array([int(dXclass)], dtype=np.uint64)
dY = np.array([int(dYclass)], dtype=np.uint64)
dOut = np.array([int(dOutclass)], dtype=np.uint64)

args = [a, dX, dY, dOut, n]
args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

## 调用kernel
err, = cuda.cuLaunchKernel(
   kernel,
   NUM_BLOCKS,  # grid x dim
   1,  # grid y dim
   1,  # grid z dim
   NUM_THREADS,  # block x dim
   1,  # block y dim
   1,  # block z dim
   0,  # dynamic shared memory
   stream,  # stream
   args.ctypes.data,  # kernel arguments，指向指针的指针
   0,  # extra (ignore)
)

err, = cuda.cuMemcpyDtoHAsync(
   hOut.ctypes.data, dOutclass, bufferSize, stream
)
err, = cuda.cuStreamSynchronize(stream)


# 资源释放
err, = cuda.cuStreamDestroy(stream)
err, = cuda.cuMemFree(dXclass)
err, = cuda.cuMemFree(dYclass)
err, = cuda.cuMemFree(dOutclass)
err, = cuda.cuModuleUnload(module)
err, = cuda.cuCtxDestroy(context)
```


# 参考资料

1. https://nvidia.github.io/cuda-python/overview.html
2. 
