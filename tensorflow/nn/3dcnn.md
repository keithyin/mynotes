#conv3d
```python
tf.nn.conv3d(input, filter, strides, padding, name=None)

Computes a 3-D convolution given 5-D input and filter tensors.

In signal processing, cross-correlation is a measure of similarity of two waveforms as a function of a time-lag applied to one of them. This is also known as a sliding dot product or sliding inner-product.

Our Conv3D implements a form of cross-correlation.

Args:

input: A Tensor. Must be one of the following types: float32, float64, int64, int32, uint8, uint16, int16, int8, complex64, complex128, qint8, quint8, qint32, half. Shape [batch, in_depth, in_height, in_width, in_channels].
filter: A Tensor. Must have the same type as input. Shape [filter_depth, filter_height, filter_width, in_channels, out_channels]. in_channels must match between input and filter.
strides: A list of ints that has length >= 5. 1-D tensor of length 5. The stride of the sliding window for each dimension of input. Must have strides[0] = strides[4] = 1.
padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
name: A name for the operation (optional).
Returns:

A Tensor. Has the same type as input.
```
这是官方给的解释,还不如conv2d解释的详细呢,至少在介绍conv2d的时候还给了公式.
**和conv2d对比一下:**
- 在input的shape中多了个 in_depth(代表一个sample输入几个帧,每帧代表一个图片).
- filter的shape也多个 filter_depth.在conv2d中, filter_height, filter_height构成感受眼的大小.在conv3d中,由filter_depth,filter_height,filter_width构成了感受眼的大小
- strides中也多了一维,[strides_batch,strides_depth,strides_height,strides_width,strides_channel],
**和conv2d相同的地方**
- 虽然多了一维,但是参数表示的意思和conv2d时是一样的,in_channels依旧是代表输入图片的channels,(e.g.RGB图像的in_channels还是3)
- out_channels依旧单个图片的out_channels

# Pooling
和卷积一样理解就可以了
```python
tf.nn.avg_pool3d(input, ksize, strides, padding, name=None)

Performs 3D average pooling on the input.

Args:

input: A Tensor. Must be one of the following types: float32, float64, int64, int32, uint8, uint16, int16, int8, complex64, complex128, qint8, quint8, qint32, half. Shape [batch, depth, rows, cols, channels] tensor to pool over.
ksize: A list of ints that has length >= 5. 1-D tensor of length 5. The size of the window for each dimension of the input tensor. Must have ksize[0] = ksize[4] = 1.
strides: A list of ints that has length >= 5. 1-D tensor of length 5. The stride of the sliding window for each dimension of input. Must have strides[0] = strides[4] = 1.
padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
name: A name for the operation (optional).
Returns:

A Tensor. Has the same type as input. The average pooled output tensor.

```
```python
tf.nn.max_pool3d(input, ksize, strides, padding, name=None)

Performs 3D max pooling on the input.

Args:

input: A Tensor. Must be one of the following types: float32, float64, int64, int32, uint8, uint16, int16, int8, complex64, complex128, qint8, quint8, qint32, half. Shape [batch, depth, rows, cols, channels] tensor to pool over.
ksize: A list of ints that has length >= 5. 1-D tensor of length 5. The size of the window for each dimension of the input tensor. Must have ksize[0] = ksize[4] = 1.
strides: A list of ints that has length >= 5. 1-D tensor of length 5. The stride of the sliding window for each dimension of input. Must have strides[0] = strides[4] = 1.
padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
name: A name for the operation (optional).
Returns:

A Tensor. Has the same type as input. The max pooled output tensor.

```
