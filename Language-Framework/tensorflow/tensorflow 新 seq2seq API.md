# tensorflow 新 seq2seq API

这里主要考虑的是 decoder部分，因为训练的时候和推断的时候具有不一致性。



`tensorflow`在此处的抽象为：

* `Decoder`: 实现了 解码时候，单个 `step` 的计算
  * `BasicDecoder` : 将 `rnn_cell, helper`封装起来，用于解码，`cell, helper, initial_state, output_layer=None`
  * `BeamSearchDecoder`
* `DecoderOutput`
  * `BasicDecoderOutput`
  * `BeamSearchDecoderOutput`
  * `FinalBeamSearchDecoderOutput`
* `DecoderState`
  * `BeamSearchDecoderState`
* `Helper`: 负责 `decoder` 的输入，不考虑`Attention机制`的时候，用来解码的 `rnn_cell` 就两个部分输入，`state, inputs`. `Helper` 就是负责计算 下一个时间步 `rnn_cell` 的输入的
  * `Helper` : 
  * `CustomHelper`
  * `TrainingHelper` : 将 `inputs, sequences_length` 封装起来，其中 `inputs` 是 `decoder_cell`的输入 `embedding` 
  * `InferenceHelper`
  * `GreedyEmbeddingHelper`
  * `SampleEmbeddingHelper`
  * `ScheduledEmbeddingTrainingHelper`
  * `ScheduledOutputTrainingHelper`
* `AttentionMechanism` : 封装 `attention_state` ，计算当前时间步的 `alignments` 。
  * `BahdanauAttention`
  * `LuongAttention`
  * `BahdanauMonotonicAttention`
  * `LuongMonotonicAttention`
* `AttentionWrapper` : 封装 `RNNCell, AttentionMechanism` 得到一个新的 `decoder_cell`。当加入`Attention机制`的时候，原始`rnn_cell` 的输入会有所改变，但是`tf` 的选择并不是扩充`Helper`的功能，而是引入`AttentionWrapper` 将原始 `rnn_cell` 和 `attention机制`封装起来，变为一个复杂一些的 `rnn_cell/decoder_cell`。
* `AttentionWrapperState`



在写seq2seq代码时：

* 训练时：
  * `decoder` 每一时间步的输入，和 `decoder` 序列的长度，由`TrainHelper`封装，`TrainHelper` 负责 `decoder` 下一个时间步的输入
  * 使用`attention` 机制的时候，`AttentionWrapper` 将 `AttentionMechnism,RNNCell` 封装程一个新的`cell`



## 封装层级

* `Helper(embedding_inputs, seq_lengths)`, `RNNCell(num_units)` , `AttentionMechism(memory)`
* `AttentionWrapper(attention_mechism, rnn_cell)->RNNCell`
* `Decoder(rnn_cell, helper)->decoder_cell`
* `dynamic_deocde(decoder_cell)`



```python
# Decoder
def step(self, time, inputs, state, name=None):
  """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
  with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
    cell_outputs, cell_state = self._cell(inputs, state)
    if self._output_layer is not None:
      cell_outputs = self._output_layer(cell_outputs)
      sample_ids = self._helper.sample(
        time=time, outputs=cell_outputs, state=cell_state)
      (finished, next_inputs, next_state) = self._helper.next_inputs(
        time=time,
        outputs=cell_outputs,
        state=cell_state,
        sample_ids=sample_ids)
      outputs = BasicDecoderOutput(cell_outputs, sample_ids)
      return (outputs, next_state, next_inputs, finished)
```







```python
attention_states = tf.transpose(encoder_outputs, [1, 0, 2])

# Create an attention mechanism
attention_mechanism = tf.contrib.seq2seq.LuongAttention(
    num_units, attention_states,
    memory_sequence_length=source_sequence_length)

# 得到的还是单 decoder_cell
decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
    decoder_cell, attention_mechanism,
    attention_layer_size=num_units)

# Replicate encoder infos beam_width times
decoder_initial_state = tf.contrib.seq2seq.tile_batch(
    encoder_state, multiplier=hparams.beam_width)

# 还可以在上面套一个 beam-search
# Define a beam-search decoder
decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell=decoder_cell,
        embedding=embedding_decoder,
        start_tokens=start_tokens,
        end_token=end_token,
        initial_state=decoder_initial_state,
        beam_width=beam_width,
        output_layer=projection_layer,
        length_penalty_weight=0.0,
        coverage_penalty_weight=0.0)

# Dynamic decoding
outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
```

