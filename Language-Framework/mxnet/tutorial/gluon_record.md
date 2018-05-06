# mxnet - autograd.record()

`mxnet0.11` 已经提供了动态图的支持。`mxnet` 使用 `with autograd.record()` 来记录计算图的定义，本文是来介绍 `with autograd.record()` 在底层做了什么。



先看 `autograd.record()` 的源码：

```python
def record(train_mode=True): #pylint: disable=redefined-outer-name
    """Returns an autograd recording scope context to be used in 'with' statement
    and captures code that needs gradients to be calculated.

    .. note:: When forwarding with train_mode=False, the corresponding backward
              should also use train_mode=False, otherwise gradient is undefined.

    Example::

        with autograd.record():
            y = model(x)
            backward([y])
        metric.update(...)
        optim.step(...)

    Parameters
    ----------
    train_mode: bool, default True
        Whether the forward pass is in training or predicting mode. This controls the behavior
        of some layers such as Dropout, BatchNorm.
    """
    return _RecordingStateScope(True, train_mode)
```

可见，这个函数只为了返回一个对象 `_RecordingStateScope`, 再来看一下这个对象是什么？

```python
class _RecordingStateScope(object):
    """Scope for managing training state.

    Example::

        with _RecordingStateScope(True, True):
            y = model(x)
            backward([y])

    """
    def __init__(self, is_record, train_mode): #pylint: disable=redefined-outer-name
        self._enter_is_record = is_record
        self._enter_train_mode = train_mode
        self._prev_is_record = None
        self._prev_train_mode = None
	
    # 定义了进入 with 块的时候做些什么。
    def __enter__(self):
        # 设置当前的状态，将之前的状态保存起来
        if self._enter_is_record is not None:
            self._prev_is_record = set_recording(self._enter_is_record)
        if self._enter_train_mode is not None:
            self._prev_train_mode = set_training(self._enter_train_mode)
    # 退出 with 块时做些什么
    def __exit__(self, ptype, value, trace):
        # 将状态恢复到之前
        if self._enter_is_record is not None and self._prev_is_record != self._enter_is_record:
            set_recording(self._prev_is_record)
        if self._enter_train_mode is not None and self._prev_train_mode != self._enter_train_mode:
            set_training(self._prev_train_mode)
```



关于 `set_recording()`

```python
def set_recording(is_recording): #pylint: disable=redefined-outer-name
    """Set status to recording/not recording. When recording, graph will be constructed
    for gradient computation.

    Parameters
    ----------
    is_recording: bool

    Returns
    -------
    previous state before this set.
    """
    prev = ctypes.c_int()
    check_call(_LIB.MXAutogradSetIsRecording(
        ctypes.c_int(is_recording), ctypes.byref(prev)))
    return bool(prev.value)
```

关于 `set_training()`

```python
def set_training(train_mode): #pylint: disable=redefined-outer-name
    """Set status to training/predicting. This affects ctx.is_train in operator
    running context. For example, Dropout will drop inputs randomly when
    train_mode=True while simply passing through if train_mode=False.

    Parameters
    ----------
    train_mode: bool

    Returns
    -------
    previous state before this set.
    """
    prev = ctypes.c_int()
    check_call(_LIB.MXAutogradSetIsTraining(
        ctypes.c_int(train_mode), ctypes.byref(prev)))
    return bool(prev.value)
```



可见，在 后端，有两个参数来记录着当前的 `record，training`状态。

