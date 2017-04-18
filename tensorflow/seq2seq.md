# seq2seq.py
[源码地址](https://github.com/tensorflow/tensorflow/blob/c8a45a8e236776bed1d14fd71f3b6755bd63cc58/tensorflow/python/ops/seq2seq.py)
引包: `from tensorflow.python.ops import seq2seq`
调用方法:`seq2seq.func_name() or tf.nn.seq2seq.func_name()`
### sequence_loss_by_example()

```python
def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
"""
logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
targets: List of 1D batch-sized int32 Tensors of the same length as logits.
weights: List of 1D batch-sized float-Tensors of the same length as logits.
average_across_timesteps: If set, divide the returned cost by the total
      label weight.
softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch to be used instead of the standard softmax (the default if this is None).
name: Optional name for this operation, default: "sequence_loss_by_example".
Returns:
  1D batch-sized float Tensor: The log-perplexity for each sequence.
"""
```
- logits: [2D_tensor_1, 2D_tensor_2, ..., 2D_tensor_m]
- targets: [1D_tensor_1, 1D_tensor_2, ..., 1D_tensor_m]
- weights: [1D_tensor_1, 1D_tensor_2, ..., 1D_tensor_m]
- return : [batch-sized] (average_across_timesteps=True,返回word的log perplexity. False,返回sentense 的 log perplexity)(感觉源码中的注释有问题(-\_-))

**log_perplexity简单的理解成cross entropy就好, 越小越好**
**weights参数的作用:**
由于我们训练的时候,如果句子的长度没有达到最大长度,我们就会通过填充的方式来解决这种问题.但是当我们训练模型的时候,填充的部分也会有loss.所以weights参数在非填充的部分为1,填充的部分为0.然后再乘上之前的losses,就会消除掉填充部分的loss.

### sequence_loss_by_example()

```python
def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
"""
logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
targets: List of 1D batch-sized int32 Tensors of the same length as logits.
weights: List of 1D batch-sized float-Tensors of the same length as logits.
average_across_timesteps: If set, divide the returned cost by the total
      label weight.
average_across_batch: If set, divide the returned cost by the batch size.
softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
name: Optional name for this operation, defaults to "sequence_loss".
Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).
"""
```
- 参数,基本和上个函数一样
- return: 上个函数返回的是[batch_sized], 这个函数返回reduce_mean([batch_sized])
average_across_timesteps=True, average_across_batch=True: 返回的是average log-perplexity per symbol
average_across_timesteps=False, average_across_batch=True:返回的是average log-perplexity per sentense at time order
average_across_timesteps=True, average_across_batch=False:返回的是average log-perplexity per senternse at batch order
average_across_timesteps=False, average_across_batch=False:返回的是log-perplexity per batch

**对于seq2seq模型,有这么几种情况:**
- encoder输入one hot, decoder输入one hot,输出one hot.
- encoder输入ids, decoder输入ids,输出 embeding. 计算loss的话需要project 成one hot.(使用了embedding)
- encoder输入ids, decoder输入ids,输出one hot. (使用了embedding)
- 上面三种decoder输入都可以使用feed_previous
### def \_extract_argmax_and_embed()
如果需要feed_prev,需要这个函数返回的loop_function
```python
def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
"""
Get a loop_function that extracts the previous symbol and embeds it.
  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.
  Returns:
    A loop function.
"""
  def loop_function(prev, _):
"""
prev:上一个输出,shape embedding if output_projection is not None else one_hot
return: embeded_prev if output_projection is not None else one_hot
"""
```

### def rnn_decoder()
用于两种基本情况:
- decoder_inputs 输入 one_hot, 输出 one_hot
- decoder_inputs 输入 embedding, 输出 embedding

loop_function是\_extract_argmax_and_embed()返回的函数
```python
def rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None,
                scope=None):
"""
Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
"""
```
