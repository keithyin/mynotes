# record

```python
autograd.record()
```



```python
def record(train_mode=True): #pylint: disable=redefined-outer-name
    return _RecordingStateScope(True, train_mode)
```



```python
class _RecordingStateScope(object):
    def __init__(self, is_record, train_mode): #pylint: disable=redefined-outer-name
        self._enter_is_record = is_record
        self._enter_train_mode = train_mode
        self._prev_is_record = None
        self._prev_train_mode = None

    def __enter__(self):
        if self._enter_is_record is not None:
            self._prev_is_record = set_recording(self._enter_is_record)
        if self._enter_train_mode is not None:
            self._prev_train_mode = set_training(self._enter_train_mode)

    def __exit__(self, ptype, value, trace):
        if self._enter_is_record is not None and self._prev_is_record != self._enter_is_record:
            set_recording(self._prev_is_record)
        if self._enter_train_mode is not None and self._prev_train_mode != self._enter_train_mode:
            set_training(self._prev_train_mode)
```



```python
def set_recording(is_recording): #pylint: disable=redefined-outer-name
    prev = ctypes.c_int()
    check_call(_LIB.MXAutogradSetIsRecording(
        ctypes.c_int(is_recording), ctypes.byref(prev)))
    return bool(prev.value)

def set_training(train_mode): #pylint: disable=redefined-outer-name
    prev = ctypes.c_int()
    check_call(_LIB.MXAutogradSetIsTraining(
        ctypes.c_int(train_mode), ctypes.byref(prev)))
    return bool(prev.value)

def is_recording():
    curr = ctypes.c_bool()
    check_call(_LIB.MXAutogradIsRecording(ctypes.byref(curr)))
    return curr.value

def is_training():
    curr = ctypes.c_bool()
    check_call(_LIB.MXAutogradIsTraining(ctypes.byref(curr)))
    return curr.value

```



```c++
int MXAutogradSetIsRecording(int is_recording, int* prev) {
  API_BEGIN();
  *prev = Imperative::Get()->set_is_recording(static_cast<bool>(is_recording));
  API_END();
}
int MXAutogradSetIsTraining(int is_training, int* prev) {
  API_BEGIN();
  *prev = Imperative::Get()->set_is_training(static_cast<bool>(is_training));
  API_END();
}

int MXAutogradIsTraining(bool* curr) {
  API_BEGIN();
  *curr = Imperative::Get()->is_training();
  API_END();
}

int MXAutogradIsRecording(bool* curr) {
  API_BEGIN();
  *curr = Imperative::Get()->is_recording();
  API_END();
}

// 单例模式，用一个 Imrerative 对象来记录的当前的 record 和 training 状态
Imperative* Imperative::Get() {
  static Imperative inst;
  return &inst;
}
```



## 参考资料

[https://github.com/apache/incubator-mxnet/blob/759f5091eb5feae68ffc5914d6cf8798dd2af6d7/src/imperative/imperative.cc#L32](https://github.com/apache/incubator-mxnet/blob/759f5091eb5feae68ffc5914d6cf8798dd2af6d7/src/imperative/imperative.cc#L32)

[https://github.com/apache/incubator-mxnet/blob/974ac1758765291337129f546434993376eae687/src/c_api/c_api_ndarray.cc#L292](https://github.com/apache/incubator-mxnet/blob/974ac1758765291337129f546434993376eae687/src/c_api/c_api_ndarray.cc#L292)



