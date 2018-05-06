# mxnet convolution summary



mxnet 的 卷积操作是根据 输入数据的维度来确定是 n-D 卷积。

## Convolution

`mxnet.symbol.Convolution(data=None, weight=None, bias=None, kernel=_Null, stride=_Null, dilate=_Null, pad=_Null, num_filter=_Null, num_group=_Null, workspace=_Null, no_bias=_Null, cudnn_tune=_Null, cudnn_off=_Null, layout=_Null, name=None, attr=None, out=None, **kwargs)`

[地址](http://mxnet.io/api/python/symbol.html#mxnet.symbol.Convolution)

> Compute *N*-D convolution on *(N+2)*-D input. 1-D 卷积，2-D卷积，3-D卷积

2-D 卷积情况：

* **data** : `(batch_size, channel, width, height)`
* **weight** : `(num_filter, channel, kernel[0], kernel[1])`
* **bias** : `(num_filter,)`
* **out** : `(batch_size, num_filter, out_height, out_width)`

**如何计算 out_height**

$floor\Biggr(\frac{x+2*p-\Bigr(2*(d-1)+1\Bigr)}{stride}\Biggr)+1$



- **stride** (*Shape(tuple), optional, default=()*) – convolution stride: (h, w) or (d, h, w)
- **dilate** (*Shape(tuple), optional, default=()*) – convolution dilate: (h, w) or (d, h, w)
- **pad** (*Shape(tuple), optional, default=()*) – pad for convolution: (h, w) or (d, h, w)
- **num_filter** (*int (non-negative), required*) – convolution filter(channel) number
- **num_group** (*int (non-negative), optional, default=1*) – Number of group partitions. Equivalent to slicing input into num_group partitions, apply convolution on each, then concatenate the results
- **workspace** (*long (non-negative), optional, default=1024*) – Maximum tmp workspace allowed for convolution (MB).
- **no_bias** (*boolean, optional, default=False*) – Whether to disable bias parameter.
- **cudnn_tune** (*{None, 'fastest', 'limited_workspace', 'off'},optional, default='None'*) – Whether to pick convolution algo by running performance test. Leads to higher startup time but may give faster speed. Options are: ‘off’: no tuning ‘limited_workspace’: run test and pick the fastest algorithm that doesn’t exceed workspace limit. ‘fastest’: pick the fastest algorithm and ignore workspace limit. If set to None (default), behavior is determined by environment variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off, 1 for limited workspace (default), 2 for fastest.
- **cudnn_off** (*boolean, optional, default=False*) – Turn off cudnn for this layer.
- **layout** (*{None, 'NCDHW', 'NCHW', 'NDHWC', 'NHWC'},optional, default='None'*) – Set layout for input, output and weight. Empty for default layout: NCHW for 2d and NCDHW for 3d.
- **name** (*string, optional.*) – Name of the resulting symbol.



## Deconvolution

`mxnet.symbol.Deconvolution(data=None, weight=None, bias=None, kernel=_Null, stride=_Null, dilate=_Null, pad=_Null, adj=_Null, target_shape=_Null, num_filter=_Null, num_group=_Null, workspace=_Null, no_bias=_Null, cudnn_tune=_Null, cudnn_off=_Null, layout=_Null, *name=None*, attr=None, out=None, **kwargs)`

> 只能用于 2-D 反卷积

**从正向卷积的方向去考虑就可以**

- **data** ([*Symbol*](http://mxnet.io/api/python/symbol.html#mxnet.symbol.Symbol)) – 输入 tensor
- **weight** ([*Symbol*](http://mxnet.io/api/python/symbol.html#mxnet.symbol.Symbol)) – 表示 kernel
- **bias** ([*Symbol*](http://mxnet.io/api/python/symbol.html#mxnet.symbol.Symbol)) – Bias added to the result after the deconvolution operation.
- **kernel** (*Shape(tuple), required*) – Deconvolution kernel size: (h, w) or (d, h, w). This is same as the kernel size used for the corresponding convolution
- **stride** (*Shape(tuple), optional, default=()*) – The stride used for the **corresponding convolution**: (h, w) or (d, h, w).
- **dilate** (*Shape(tuple), optional, default=()*) – Dilation factor for each dimension of the input: (h, w) or (d, h, w).
- **pad** (*Shape(tuple), optional, default=()*) – The amount of implicit zero padding added **during convolution** for each dimension of the input: (h, w) or (d, h, w). `(kernel-1)/2` is usually a good choice. If target_shape is set, pad will be ignored and a padding that will generate the target shape will be used.
- **adj** (*Shape(tuple), optional, default=()*) – Adjustment for output shape: (h, w) or (d, h, w). If target_shape is set, adj will be ignored and computed accordingly.
- **target_shape** (*Shape(tuple), optional, default=()*) – Shape of the output tensor: (h, w) or (d, h, w).
- **num_filter** (*int (non-negative), required*) – Number of output filters.
- **num_group** (*int (non-negative), optional, default=1*) – Number of groups partition.
- **workspace** (*long (non-negative), optional, default=512*) – Maximum temporal workspace allowed for deconvolution (MB).
- **no_bias** (*boolean, optional, default=True*) – Whether to disable bias parameter.
- **cudnn_tune** (*{None, 'fastest', 'limited_workspace', 'off'},optional, default='None'*) – Whether to pick convolution algorithm by running performance test.
- **cudnn_off** (*boolean, optional, default=False*) – Turn off cudnn for this layer.
- **layout** (*{None, 'NCDHW', 'NCHW', 'NCW', 'NDHWC', 'NHWC'},optional, default='None'*) – Set layout for input, output and weight. Empty for default layout, NCW for 1d, NCHW for 2d and NCDHW for 3d.
- **name** (*string, optional.*) – Name of the resulting symbol.