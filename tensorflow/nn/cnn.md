#tenforflow:CNN
##常用函数
###卷积函数
1.tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
2.tf.nn.depthwise_conv2d(input, filter, strides, padding, name=None)
3.tf.nn.separable_conv2d(input, depthwise_filter, pointwise_filter, strides, padding, name=None)
4.tf.nn.atrous_conv2d(value, filters, rate, padding, name=None)

####tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
```python
def conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None):
#input:shape [batch_size, in_height, in_width, in_channels].channels的意思是就是，如果你输入的是RGB图像，channel就是3
#filter:shape [filter_height, filter_width, in_channels, out_channels], out_channels表示输出几张feature map
#strides:A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.
#stride:stride=[1,h_stride,v_stride,1] ，strides[0]=strides[3]=1!!!!!!第0个是batch中的样本，第三个是channel
#padding:两种形式"VALID"和"SAME","VALID"不会去补0,"SAME"和"VALID"一样运算，不够的时候，会补0，不知为啥tensorflow没有"FULL"
#输出:[batch_size, out_height, out_width, out_channels]
```
![Image_from_Skype2.jpg](/home/keith/Downloads/Image_from_Skype2.jpg)
**每三个filter看作一组，每组中的权值不是共享的，组之间也不是共享的**
<center>图一：conv2d的输入输出</center>
```python
def depthwise_conv2d(input, filter, strides, padding, name=None):
#input:shape [batch_size, in_height, in_width, in_channels]
#filter:shape [filter_height, filter_width, in_channels, channel_multiplier]
#strides:同上
#padding:同上
#return:shape [batch, out_height, out_width, in_channels * channel_multiplier]
```
**各filter之间权值不共享**
![Image_from_Skype3.jpg](/home/keith/Downloads/Image_from_Skype3.jpg)
<center>图二：depthwise_conv2d的输入输出</center>
**剩下两个函数暂时还没研究**

###POOLING函数
1.tf.nn.avg_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
2.tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
3.tf.nn.max_pool_with_argmax(input, ksize, strides, padding, Targmax=None, name=None)
```python
def avg_pool(value, ksize, strides, padding, data_format='NHWC', name=None):
#value:shape [batch, height, width, channels]
#ksize:A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
#strides:A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.一般为[1, h_stride, v_stride, 1]
#return: [batch, out__height, out_width, out_channels]

#max_pool与avg_pool相似
```
