# tensorflow 初始化模型参数时做了什么

当我们训练自己的神经网络的时候，无一例外的就是都会加上一句 `sess.run(tf.global_variables_initializer())` ，这行代码的官方解释是 初始化模型的参数。那么，它到底做了些什么？



**一步步看源代码：**

* `global_variables_initializer` 返回一个用来初始化 计算图中 所有`global variable`的 `op`。
  *  这个`op` 到底是啥，还不清楚。
  * 函数中调用了 `variable_initializer()` 和 `global_variables()`
* `global_variables()` 返回一个 `Variable list` ，里面保存的是 `gloabal variables`。
* `variable_initializer()` 将 `Variable list` 中的所有 `Variable` 取出来，将其 `variable.initializer` 属性做成一个 `op group`。
* 然后看 `Variable` 类的源码可以发现， `variable.initializer` 就是一个 `assign op`。



**所以：** `sess.run(tf.global_variables_initializer())` 只是对所有的 `Variable` 做了个 `assign` 操作，这就是初始化参数的本来面目。

```python
def global_variables_initializer():
  """Returns an Op that initializes global variables.
  Returns:
    An Op that initializes global variables in the graph.
  """
  return variables_initializer(global_variables())

def global_variables():
  """Returns global variables.
  Returns:
    A list of `Variable` objects.
  """
  return ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

def variables_initializer(var_list, name="init"):
  """Returns an Op that initializes a list of variables.
  Args:
    var_list: List of `Variable` objects to initialize.
    name: Optional name for the returned operation.

  Returns:
    An Op that run the initializers of all the specified variables.
  """
  if var_list:
    return control_flow_ops.group(*[v.initializer for v in var_list], name=name)
  return control_flow_ops.no_op(name=name)
```



```python
class Variable(object):
    def _init_from_args(self, ...):
        self._initializer_op = state_ops.assign(
            self._variable, self._initial_value,
            validate_shape=validate_shape).op
    @property
    def initializer(self):
        """The initializer operation for this variable."""
        return self._initializer_op
```



