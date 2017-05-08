# tensorflow buckets

`tensorflow`的编码原则是,先构建计算图,然后再去执行计算图(`sess.run()`).这就会导致一个问题,我们无法在运行的过程中动态的更改图的结构.**我们能做的就是,先构建出完整的图,然后可以去执行其子图.** `tensorflow`中的`bucket`就是基于这么一种思想.
## seq2seq简单介绍
在`seq2seq`场景中,输入和输出的`sequence`的长度往往是变长的.假设在`train set`中, `encoder sequence`的长度最长为`100`,`decoder sequence`的长度最长为`150`, 那么我们往往需要将所有的`encoder sequence`给`pad`成长度100的sequence, 将所有的`decoder sequence`给`pad`成长度150的sequence.而相应的,`encoder rnn`单元的个数至多是100,`decoder rnn` 单元的个数至多是150.你可能会纠结为什么是至多,因为从编码的角度来讲,如果`rnn` 单元的个数少于100, 那么序列中的最后几个数据就不用去考虑, 想反,如果`rnn`的单元个数多,那么就会存在某些 `rnn`单元的输入没有进行初始话,会出错.
```{mermaid}
graph LR
  subgraph encoder
  t1 --> t2
  t2 --> t3
  t3 --> ...
  ... --> tm
  end
  subgraph decoder
  t_1 --> t_2
  t_2 --> t_3
  t_3 --> _...
  _... --> t_n
  end
```
`encoder`和`decoder`的基本形式如图,两者以某种形式配合起来,就会构成`seq2seq`结构.其中`m` `n`分别代表`encoder sequence`和 `decoder sequence`的最大长度.

## 为什么需要bucket
`bucket`就是一种编码思想，`bucket`的存在是为了减小计算量,从而可以减少模型的训练时间。当然，使用`dynamic_rnn`或`rnn`这两个接口也可以减少运算时间。`bucket`是用在使用在`cell(input,state)`这种古老的方法上的。
关于`bucket`的源码是在[https://github.com/tensorflow/tensorflow/blob/27711108b5fce2e1692f9440631a183b3808fa01/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py](https://github.com/tensorflow/tensorflow/blob/27711108b5fce2e1692f9440631a183b3808fa01/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py)。
要实现`bucket`：
1. 对`train set`:要对`sequence`的长度聚类，确保如何分配`bucket`。
2. 数据依旧要填充到最大长度
3. 对每个`buckets`都要建立一个一个模型，但是模型都是共享变量的
4. 对每个模型都要都要计算`loss`，保存到list中
5. 训练的时候，最小化对应`bucket`的loss
知道这些，看代码就容易多了
```python
def model_with_buckets(encoder_inputs,
                       decoder_inputs,
                       targets,
                       weights,
                       buckets,
                       seq2seq,
                       softmax_loss_function=None,
                       per_example_loss=False,
                       name=None):
  """Create a sequence-to-sequence model with support for bucketing.
  The seq2seq argument is a function that defines a sequence-to-sequence model,
  e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))
  Args:
    encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
    decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
    targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
    weights: List of 1D batch-sized float-Tensors to weight the targets.
    buckets: A list of pairs of (input size, output size) for each bucket.
    seq2seq: A sequence-to-sequence model function; it takes 2 input that
      agree with encoder_inputs and decoder_inputs, and returns a pair
      consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
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
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  with ops.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=True if j > 0 else None):
        bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                    decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        if per_example_loss:
          losses.append(
              sequence_loss_by_example(
                  outputs[-1],
                  targets[:bucket[1]],
                  weights[:bucket[1]],
                  softmax_loss_function=softmax_loss_function))
        else:
          losses.append(
              sequence_loss(
                  outputs[-1],
                  targets[:bucket[1]],
                  weights[:bucket[1]],
                  softmax_loss_function=softmax_loss_function))

  return outputs, losses
```
**片段分析**
```python
if len(encoder_inputs) < buckets[-1][0]:
  raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                   "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
if len(targets) < buckets[-1][1]:
  raise ValueError("Length of targets (%d) must be at least that of last"
                   "bucket (%d)." % (len(targets), buckets[-1][1]))
if len(weights) < buckets[-1][1]:
  raise ValueError("Length of weights (%d) must be at least that of last"
                   "bucket (%d)." % (len(weights), buckets[-1][1]))
#可以看出，输入数据必须填充为最大长度
```

```python
for j, bucket in enumerate(buckets):#对每个bucket创建一个模型
  with variable_scope.variable_scope(
      variable_scope.get_variable_scope(), reuse=True if j > 0 else None):
    bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                decoder_inputs[:bucket[1]])
    outputs.append(bucket_outputs)
    if per_example_loss:
      losses.append(
          sequence_loss_by_example(
              outputs[-1],
              targets[:bucket[1]],
              weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))
    else:#所有模型的loss都被保存起来
      losses.append(
          sequence_loss(
              outputs[-1],
              targets[:bucket[1]],
              weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))
```
**总结**
使用`tensorflow`编码的时候，分为`构建计算图`和`执行计算图`部分，上面的代码是用于`构建计算图`，我们对不同的`bucket`构建了不同的`计算图`。在`执行计算图`阶段，`tensorflow`只会运算`子图`。假设我们有一个`minibatch`数据，与这批数据最相近的`bucket`的id是`3`，那么在训练的时候，我们只需要 最小化`losses[3]`就可以了。
