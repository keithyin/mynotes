# seq2seq Model
[源码地址](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/seq2seq_model.py)
## 调用外部的函数介绍
### tf.sampled_softmax_loss()
tf.sampled_softmax_loss()中调用了_compute_sampled_logits() [关于__compute_sampled_logits()](http://blog.csdn.net/u012436149/article/details/52848013)
```python
#此函数和nce_loss是差不多的, 取样求loss
def sampled_softmax_loss(weights, #[num_classes, dim]
                         biases,  #[num_classes]
                         inputs,  #[batch_size, dim]
                         labels,  #[batch_size, num_true]
                         num_sampled,
                         num_classes,
                         num_true=1,
                         sampled_values=None,
                         remove_accidental_hits=True,
                         partition_strategy="mod",
                         name="sampled_softmax_loss"):
#return: [batch_size]
```
**关于参数labels:一般情况下，num_true为1, labels的shpae为[batch_size, 1]。假设我们有1000个
类别， 使用one_hot形式的label的话， 我们的labels的shape是[batch_size, num_classes]。显然，如果
num_classes非常大的话，会影响计算性能。所以，这里采用了一个简化的方式,即：使用3代表了[0,0,0,1,0....]**
### tf.nn.seq2seq.embedding_attention_seq2seq（）
创建了input embedding matrix 和 output embedding matrix
```python
def embedding_attention_seq2seq(encoder_inputs, #[T， batch_size]
                                decoder_inputs, #[out_T， batch_size]
                                cell,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                num_heads=1, #只采用一个read head
                                output_projection=None,
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
#output_projection: (W, B) W:[output_size, num_decoder_symbols]
#B: [num_decoder_symbols]                    
```
(1)这个函数创建了一个inputs 的 embedding matrix.
(2)计算了encoder的 output，并保存起来，用于计算attention
```python
encoder_cell = rnn_cell.EmbeddingWrapper(
      cell, embedding_classes=num_encoder_symbols,
      embedding_size=embedding_size)# 创建了inputs的 embedding matrix
  encoder_outputs, encoder_state = rnn.rnn(
      encoder_cell, encoder_inputs, dtype=dtype) #return [T ，batch_size，size]
```
（3）生成attention states
```python
  top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                for e in encoder_outputs]  # T * batch_size * 1 * size
  attention_states = array_ops.concat(1, top_states) # batch_size*T*size
```
（4）剩下的工作交给embedding_attention_decoder,embedding_attention_decoder中创建了decoder的embedding matrix
```python
# Decoder.
  output_size = None
  if output_projection is None:
    cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
    output_size = num_decoder_symbols

  if isinstance(feed_previous, bool):
    return embedding_attention_decoder(
        decoder_inputs,
        encoder_state,
        attention_states,
        cell,
        num_decoder_symbols,
        embedding_size,
        num_heads=num_heads,
        output_size=output_size,
        output_projection=output_projection,
        feed_previous=feed_previous,
        initial_state_attention=initial_state_attention)
```
#### tf.nn.rnn_cell.EmbeddingWrapper()
**embedding_attention_seq2seq中调用了这个类**
使用了这个类之后，rnn 的inputs就可以是[batch_size]了，里面保存的是word的id。
此类就是在 cell 前 加了一层embedding
```python
class EmbeddingWrapper(RNNCell):
  def __init__(self, cell, embedding_classes, embedding_size, initializer=None):
  def __call__(self, inputs, state, scope=None):#生成embedding矩阵[embedding_classes,embedding_size]
  #inputs: [batch_size, 1]
  #return : (output, state)
```
#### tf.nn.rnn_cell.OutputProgectionWrapper()
**将rnn_cell的输出映射成想要的维度**
```python
class OutputProjectionWrapper(RNNCell):
  def __init__(self, cell, output_size): # output_size:映射后的size
  def __call__(self, inputs, state, scope=None):
#init 返回一个带output projection的 rnn_cell
```
#### tf.nn.seq2seq.embedding_attention_decoder()
```python
#生成embedding matrix ：[num_symbols, embedding_size]
def embedding_attention_decoder(decoder_inputs, # T*batch_size
                                initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
#核心代码
  embedding = variable_scope.get_variable("embedding",
                                          [num_symbols, embedding_size])  #output embedding
  loop_function = _extract_argmax_and_embed(
      embedding, output_projection,
      update_embedding_for_previous) if feed_previous else None
  emb_inp = [
      embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
  return attention_decoder(
      emb_inp,
      initial_state,
      attention_states,
      cell,
      output_size=output_size,
      num_heads=num_heads,
      loop_function=loop_function,
      initial_state_attention=initial_state_attention)
```
可以看到，**此函数先为 decoder symbols 创建了一个embedding矩阵**。然后定义了loop_function。
emb_in是embedded input ：[T, batch_size, embedding_size]
函数的主要工作还是交给了attention_decoder()
##### tf.nn.attention_decoder()
```python
def attention_decoder(decoder_inputs, #[T, batch_size, input_size]
                      initial_state,  #[batch_size, cell.states]
                      attention_states, #[batch_size , attn_length , attn_size]
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):

```
论文中，在计算attention distribution的时候，提到了三个公式
$$(1) u_i^t = v^T*tanh(W_1*h_i + W_2*d_t) $$
$$(2) s_i^t = softmax(a_i^t) $$
$$(3) d' = \sum_{i=1}^{T_A} s_i^t * h_i$$
其中，$W_1$维度是[attn_vec_size, size], $h_i$:[size,1]，**我们日常表示输入数据，都是用列向量表示，但是在tensorflow中，趋向用行向量表示**。在这个函数中，为了计算 $ W_1*h_i $， 使用了卷积的方式。
```python
hidden = array_ops.reshape(
      attention_states, [-1, attn_length, 1, attn_size]) #[batch_size * T * 1 * input_size]
  hidden_features = []
  v = []
  attention_vec_size = attn_size  # Size of query vectors for attention.
  for a in xrange(num_heads):
    k = variable_scope.get_variable("AttnW_%d" % a,
                                    [1, 1, attn_size, attention_vec_size])
    hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
    v.append(
        variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size])) #attention_vec_size = attn_size
```
使用conv2d之后，返回的tensor的形状是[batch_size, attn_length, 1, attention_vec_size]
此函数是这么求 $W_2*d_t$ 和 $s_i$的。
```python
     y = linear(query, attention_vec_size, True)
     y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
     # Attention mask is a softmax of v^T * tanh(...).
     s = math_ops.reduce_sum(
         v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3]) #[batch_size, attn_length, 1, attn_size]
     a = nn_ops.softmax(s) #s" [batch_size * attn_len]
     # Now calculate the attention-weighted vector d.
     d = math_ops.reduce_sum(
         array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
         [1, 2])
     ds.append(array_ops.reshape(d, [-1, attn_size]))
```
 $y = W_2*d_t, d = d'$
#### def rnn()
```python
from tensorflow.python.ops import rnn
rnn.rnn()
def rnn(cell, inputs, initial_state=None, dtype=None,
        sequence_length=None, scope=None):
#inputs: A length T list of inputs, each a `Tensor` of shape`[batch_size, input_size]`
#sequence_length: [batch_size], 指定sample 序列的长度
#return : (outputs, states), outputs: T*batch_size*output_size. states:batch_size*state
```
## seq2seqModel
- 创建映射参数 proj_w, proj_b
- 声明：sampled_loss，看了word2vec的就会理解
- 声明：seq2seq_f()，构建了inputs的embedding和outputs的embedding，进行核心计算
- 使用model_with_buckets(),model_with_buckets中调用了seq2seq_f和 sampled_loss
### model_with_buckets()
调用方法 tf.nn.seq2seq.model_with_buckets()
```python
def model_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                       buckets, seq2seq, softmax_loss_function=None,
                       per_example_loss=False, name=None):
"""Create a sequence-to-sequence model with support for bucketing.
The seq2seq argument is a function that defines a sequence-to-sequence model,
e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))
Args:
encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
weights: List of 1D batch-sized float-Tensors to weight the targets.
buckets: A list of pairs of (input size, output size) for each bucket.
seq2seq: A sequence-to-sequence model function; it takes 2 input that agree with encoder_inputs and decoder_inputs, and returns a pair consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
to be used instead of the standard softmax (the default if this is None).
per_example_loss: Boolean. If set, the returned loss will be a batch-sized
tensor of losses for each sequence in the batch. If unset, it will be
a scalar with the averaged loss from all examples.
name: Optional name for this operation, defaults to "model_with_buckets".
Returns:
A tuple of the form (outputs, losses), where:
outputs: The outputs for each bucket. Its j'th element consists of a list
of 2D Tensors. The shape of output tensors can be either
[batch_size x output_size] or [batch_size x num_decoder_symbols]
depending on the seq2seq model used.
losses: List of scalar Tensors, representing losses for each bucket, or,
if per_example_loss is set, a list of 1D batch-sized float Tensors.
Raises:
ValueError: If length of encoder_inputsut, targets, or weights is smaller
than the largest (last) bucket.
"""
```
**记住，tensorflow的编码方法是：先构图，再训练。训练是根据feed确定的**
