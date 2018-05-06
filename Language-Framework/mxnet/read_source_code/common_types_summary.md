# mxnet dtypes

这个在 `base.py` 中。

```python
mx_uint = ctypes.c_uint
mx_float = ctypes.c_float
mx_float_p = ctypes.POINTER(mx_float)
mx_real_t = np.float32
NDArrayHandle = ctypes.c_void_p
FunctionHandle = ctypes.c_void_p
OpHandle = ctypes.c_void_p
CachedOpHandle = ctypes.c_void_p
SymbolHandle = ctypes.c_void_p
ExecutorHandle = ctypes.c_void_p
DataIterCreatorHandle = ctypes.c_void_p
DataIterHandle = ctypes.c_void_p
KVStoreHandle = ctypes.c_void_p
RecordIOHandle = ctypes.c_void_p
RtcHandle = ctypes.c_void_p
CudaModuleHandle = ctypes.c_void_p
CudaKernelHandle = ctypes.c_void_p
```

