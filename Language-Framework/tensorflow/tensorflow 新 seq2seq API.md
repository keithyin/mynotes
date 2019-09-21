# tensorflow 新 seq2seq API

这里主要考虑的是 decoder部分，因为训练的时候和推断的时候具有不一致性。



`tensorflow`在此处的抽象为：

* `RNNCell`
* `TrainingHelper` : 存解码用的 `embedding`，`decoder_lengths`
* `BasicDecoder`: `decoder_cell, training_helper, encoder_state`搞到一起

* `LuongAttention` : 
* `AttentionWrapper` : 将 `attention` 和 `decoder_cell` 搞在一起
*  `BeamSearchDecoder`:



* `Decoder`
  * `BasicDecoder` : 将 `rnn_cell, helper`封装起来，用于解码，`cell, helper, initial_state, output_layer=None`
  * `BeamSearchDecoder`
* `DecoderOutput`
  * `BasicDecoderOutput`
  * `BeamSearchDecoderOutput`
  * `FinalBeamSearchDecoderOutput`
* `DecoderState`
  * `BeamSearchDecoderState`
* `Helper`
  * `Helper` : 
  * `CustomHelper`
  * `TrainingHelper` : 将 `inputs, sequences_length` 封装起来，其中 `inputs` 是 `decoder_cell`的输入 `embedding` 
  * `InferenceHelper`
  * `GreedyEmbeddingHelper`
  * `SampleEmbeddingHelper`
  * `ScheduledEmbeddingTrainingHelper`
  * `ScheduledOutputTrainingHelper`
* `Attention`
  * `BahdanauAttention`
  * `LuongAttention`
  * `BahdanauMonotonicAttention`
  * `LuongMonotonicAttention`
* `AttentionWrapper`
* `AttentionMechanism`
* `AttentionWrapperState`



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

