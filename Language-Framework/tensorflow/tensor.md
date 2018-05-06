# tensorlfow Tensor
Represents one of the outputs of an `Operation`.
  A `Tensor` is a symbolic handle to one of the outputs of an
  `Operation`. It does not hold the values of that operation's output,
  but instead provides a means of computing those values in a
  TensorFlow [`Session`](../../api_docs/python/client.md#Session).
  This class has two primary purposes:
  1. A `Tensor` can be passed as an input to another `Operation`.
     This builds a dataflow connection between operations, which
     enables TensorFlow to execute an entire `Graph` that represents a
     large, multi-step computation.
  2. After the graph has been launched in a session, the value of the
     `Tensor` can be computed by passing it to
     [`Session.run()`](../../api_docs/python/client.md#Session.run).
     `t.eval()` is a shortcut for calling
     `tf.get_default_session().run(t)`.
  In the following example, `c`, `d`, and `e` are symbolic `Tensor`
  objects, whereas `result` is a numpy array that stores a concrete value

[更多](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/api_docs/python/framework.md)
