# backward

**Variable.backward**

```python
def backward(self, gradient=None, retain_graph=None, create_graph=None, retain_variables=None):
    """Computes the gradient of current variable w.r.t. graph leaves.

    This function accumulates gradients in the leaves - you might need to
    zero them before calling it.

    Arguments:
        grad_variables (Tensor, Variable or None): Gradient w.r.t. the
            variable. If it is a tensor, it will be automatically converted
            to a Variable that is volatile unless ``create_graph`` is True.
            None values can be specified for scalar Variables or ones that
            don't require grad. If a None value would be acceptable then
            this argument is optional.
        retain_graph (bool, optional): If False, the graph used to compute
            the grads will be freed. Note that in nearly all cases setting
            this option to True is not needed and often can be worked around
            in a much more efficient way. Defaults to the value of
            ``create_graph``.
        create_graph (bool, optional): If true, graph of the derivative will
            be constructed, allowing to compute higher order derivative
            products. Defaults to False, unless ``gradient`` is a volatile
            Variable.
    """
    torch.autograd.backward(self, gradient, retain_graph, create_graph, retain_variables)
```





```python
def backward(variables, grad_variables=None, retain_graph=None, create_graph=None, retain_variables=None):
    """Computes the sum of gradients of given variables w.r.t. graph leaves.

    Arguments:
        variables (sequence of Variable): Variables of which the derivative will be
            computed.
        grad_variables (sequence of (Tensor, Variable or None)): Gradients w.r.t.
            each element of corresponding variables.  Any tensors will be
            automatically converted to Variables that are volatile unless
            ``create_graph`` is True.  None values can be specified for scalar
            Variables or ones that don't require grad. If a None value would
            be acceptable for all grad_variables, then this argument is optional.
    """
    variables = (variables,) if isinstance(variables, Variable) else tuple(variables)

    if grad_variables is None:
        grad_variables = [None] * len(variables)
    elif isinstance(grad_variables, Variable) or torch.is_tensor(grad_variables):
        grad_variables = [grad_variables]
    else:
        grad_variables = list(grad_variables)

    grad_variables, create_graph = _make_grads(variables, grad_variables, create_graph)

    if retain_variables is not None:
        if retain_graph is not None:
            raise ValueError("only one of retain_graph and retain_variables can be specified")
        retain_graph = retain_variables
        warnings.warn("retain_variables option is deprecated and will be removed in 0.3. "
                      "Use retain_graph instead.")
    elif retain_graph is None:
        retain_graph = create_graph

    Variable._execution_engine.run_backward(
        variables, grad_variables, retain_graph)
```



```python
# Variable._execution_engine = ImperativeEngine()
# imports
import torch._C._functions as _functions # <module 'torch._C._functions'>

from .object import object

class _ImperativeEngine(object):
    # no doc
    def run_backward(self, *args, **kwargs): # real signature unknown
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass
```





啊啊啊啊啊啊， 追不动了！！！！