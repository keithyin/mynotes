# tensorflow 语音识别
最近在做语音识别的项目，现在项目告一段落，就把最近碰到的东西做一个总结。
## python中关于语音处理的库
* 读取wav文件
```python
import scipy.io.wavfile as wav
fs, audio = wav.read(file_name)
```
* 对读取的音频信息求MFCC（Mel频率倒谱系数）
```python
from python_speech_features import mfcc
from python_speech_features import delta
#求MFCC
processed_audio = mfcc(audio, samplerate=fs)
#求差分（一阶，二阶）
delta1 = delta(processed_audio, 1)
delta2 = delta(processed_audio, 2)
```

## tensorflow中做语音识别会碰到的API
这个部分包括了`SparseTensor`, `sparse_tensor_to_dense`,`edit_distance`。
### SparseTensor(indices, values, dense_shape)
- indices: 一个2D的 `int64 Tensor`,shape为`(N, ndims)`,指定了`sparse tensor`中的索引, 例如: indices=[[1,3], [2,4]]说明,`dense tensor`中对应索引为`[1,3], [2,4]`位置的元素的值不为0.

- values: 一个`1D tensor`,`shape`为`(N)`用来指定索引处的值. For example, given indices=[[1,3], [2,4]], the parameter values=[18, 3.6] specifies that element [1,3] of the sparse tensor has a value of 18, and element [2,4] of the tensor has a value of 3.6.

- dense_shape: 一个1D的`int64 tensor`,形状为`ndims`,指定`dense tensor`的形状.

相对应的有一个`tf.sparse_placeholder`,如果给这个`sparse_placeholder`喂数据呢?
```python
sp = tf.sparse_placeholder(tf.int32)

with tf.Session() as sess:
  #就这么喂就可以了
  feed_dict = {sp:(indices, values, dense_shape)}

```
> `tensorflow`中目前没有API提供denseTensor->SparseTensor转换

### tf.sparse_tensor_to_dense(sp_input, default_value=0, validate_indices=True, name=None)
把一个`SparseTensor`转化为`DenseTensor`.
- sp_input: 一个`SparceTensor`.

- default_value:没有指定索引的对应的默认值.默认为0.

- validate_indices: 布尔值.如果为`True`的话,将会检查`sp_input`的`indices`的`lexicographic order`和是否有重复.

- name: 返回tensor的名字前缀.可选.

### tf.edit_distance(hypothesis, truth, normalize=True, name='edit_distance')

计算序列之间的`Levenshtein` 距离

- hypothesis: `SparseTensor`,包含序列的假设.

- truth: `SparseTensor`, 包含真实序列.

- normalize: 布尔值,如果值`True`的话,求出来的`Levenshtein`距离除以真实序列的长度. 默认为`True`

- name: `operation` 的名字,可选.

返回值:
返回值是一个`R-1`维的`DenseTensor`.包含着每个`Sequence`的`Levenshtein` 距离.

`SparseTensor`所对应的`DenseTensor`是一个多维的`Tensor`,最后一维看作序列.

## CTCloss
现在用深度学习做语音识别，基本都会在最后一层用`CTCloss`，这个`loss`自己实现起来还是有点费劲，不过，幸运的是，`tensorflow`中已经有现成的`API`了，我们只需调用即可。

### tf.nn.ctc_loss(labels, inputs, sequence_length, preprocess_collapse_repeated=False, ctc_merge_repeated=True)
此函数用来计算`ctc loss`.
- labels:是一个`int32`的`SparseTensor`, `labels.indices[i, :] == [b, t]` 表示 `labels.values[i]` 保存着`(batch b, time t)`的 `id`.

- inputs:一个`3D Tensor` `(max_time * batch_size * num_classes)`.保存着 `logits`.(通常是`RNN`接上一个线性神经元的输出)

- sequence_length: 1-D int32 向量, `size`为 `[batch_size]`. 序列的长度.此 `sequence_length` 和用在`dynamic_rnn`中的sequence_length是一致的,使用来表示`rnn`的哪些输出不是`pad`的.

- preprocess_collapse_repeated:设置为`True`的话,`tensorflow`会对输入的`labels`进行预处理,连续重复的会被合成一个.

- ctc_merge_repeated: 连续重复的是否被合成一个
返回值:
一个 1-D float Tensor, `size` 为 `[batch]`, 包含着负的 $\text{log}p$.加起来即为`batch loss`.

### tf.nn.ctc_greedy_decoder(inputs, sequence_length, merge_repeated=True)
上面的函数是用在训练过程中,专注与计算`loss`,此函数是用于`inference`过程中,用于解码.

- inputs:一个`3D Tensor` `(max_time * batch_size * num_classes)`.保存着 `logits`.(通常是`RNN`接上一个线性神经元的输出)

- sequence_length: 1-D int32 向量, `size`为 `[batch_size]`. 序列的长度.此 `sequence_length` 和用在`dynamic_rnn`中的sequence_length是一致的,使用来表示`rnn`的哪些输出不是`pad`的.

返回值:
一个`tuple (decoded, log_probabilities)`

- decoded: 一个只有一个元素的哦`list`. `decoded[0]`是一个`SparseTensor`,保存着解码的结果.
  -  decoded[0].indices: 索引矩阵,size为`(total_decoded_outputs * 2)`,每行中保存着`[batch, time ]`.
  -  decoded[0].values: 值向量,`size`为 `(total_decoded_outputs)`.向量中保存的是解码的类别.
  -  decoded[0].shape: 稠密`Tensor`的`shape`, size为`(2)`.`shape`的值为`[batch_size, max_decoded_length]`.

- log_probability: 一个浮点型矩阵`(batch_size*1)`包含着序列的log 概率.

### tf.nn.ctc_beam_search_decoder(inputs, sequence_length, beam_width=100, top_paths=1,merge_repeated=True)

另一种寻路策略。和上面那个差不多。

**知道这些，就可以使用`tensorflow`搭建一个简单的语音识别应用了。**
## 参考资料
[https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss)
[https://www.tensorflow.org/api_docs/python/tf/nn/ctc_greedy_decoder](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_greedy_decoder)
[https://www.tensorflow.org/api_docs/python/tf/nn/ctc_beam_search_decoder](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_beam_search_decoder)
[http://stackoverflow.com/questions/38059247/using-tensorflows-connectionist-temporal-classification-ctc-implementation](http://stackoverflow.com/questions/38059247/using-tensorflows-connectionist-temporal-classification-ctc-implementation)
[https://www.tensorflow.org/versions/r0.10/api_docs/python/nn/conectionist_temporal_classification__ctc_](https://www.tensorflow.org/versions/r0.10/api_docs/python/nn/conectionist_temporal_classification__ctc_)
