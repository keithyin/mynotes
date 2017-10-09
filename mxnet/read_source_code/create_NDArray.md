# mxnet 创建 NDArray

```python
nd.array()
```

```python
def array(source_array, ctx=None, dtype=None, aux_types=None):
    
    if spsp is not None and isinstance(source_array, spsp.csr.csr_matrix):
        return _sparse_array(source_array, ctx=ctx, dtype=dtype, aux_types=aux_types)
    elif isinstance(source_array, NDArray) and source_array.stype != 'default':
        return _sparse_array(source_array, ctx=ctx, dtype=dtype, aux_types=aux_types)
    else:
        return _array(source_array, ctx=ctx, dtype=dtype)
```

```python
def array(source_array, ctx=None, dtype=None):
    if isinstance(source_array, NDArray):
        dtype = source_array.dtype if dtype is None else dtype
    else:
        dtype = mx_real_t if dtype is None else dtype
        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=dtype)
            except:
                raise TypeError('source_array must be array like object')
    # 先分配空间，然后初始化
    arr = empty(source_array.shape, ctx, dtype)
    arr[:] = source_array
    return arr
```

```python
def empty(shape, ctx=None, dtype=None):
    if isinstance(shape, int):
        shape = (shape, )
    if ctx is None:
        ctx = Context.default_ctx
    if dtype is None:
        dtype = mx_real_t
    # false 代表，不延迟分配空间
    return NDArray(handle=_new_alloc_handle(shape, ctx, False, dtype))
```

```python
def _new_alloc_handle(shape, ctx, delay_alloc, dtype=mx_real_t):
    """Return a new handle with specified shape and context.
	返回一个指向 NDArray 的指针
    Empty handle is only used to hold results.

    Returns
    -------
    handle
        A new empty `NDArray` handle.
    """
    hdl = NDArrayHandle()
    check_call(_LIB.MXNDArrayCreateEx(
        c_array(mx_uint, shape),
        mx_uint(len(shape)),
        ctypes.c_int(ctx.device_typeid),
        ctypes.c_int(ctx.device_id),
        ctypes.c_int(int(delay_alloc)),
        ctypes.c_int(int(_DTYPE_NP_TO_MX[np.dtype(dtype).type])),
        ctypes.byref(hdl)))
    return hdl
```

