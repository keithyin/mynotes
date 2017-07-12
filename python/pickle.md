# python pickle

> pickle 是一个 对象序列化和反序列化工具



**下列的类型可以被 pickle**

* None, True, False
* integers, floating point numbers, complex numbers
* 保存了可 pickle 对象的 tuple, list, dict
* built-in functions defined at the top level of a module
* classes that are defined at the top level of a module
* instances of such classes whose [`__dict__`](https://docs.python.org/3/library/stdtypes.html#object.__dict__) or the result of calling [`__getstate__()`](https://docs.python.org/3/library/pickle.html#object.__getstate__) is picklable (see section [Pickling Class Instances](https://docs.python.org/3/library/pickle.html#pickle-inst) for details).



> python 中，类是对象，module 也是对象，pickle 可以 序列化对象，意思就是，pickle不仅能序列化对象，而且还能序列化类。



> Note that functions (built-in and user-defined) are pickled by “fully qualified” name reference, not by value. **This means that only the function name is pickled**, along with the name of the module the function is defined in. **Neither the function’s code, nor any of its function attributes are pickled**. Thus the defining module must be importable in the unpickling environment, and the module must contain the named object, otherwise an exception will be raised. 
>
> Similarly, **classes are pickled by named reference**, so the same restrictions in the unpickling environment. Note that none of the class’s code or data is pickled, so in the following example the class attribute `attr` is not restored in the unpickling environment:
>
> **These restrictions are why picklable functions and classes must be defined in the top level of a module.**

```python
class Foo:
    attr='A class attribute'
```



> 相似的，当 pickle 类的实例的时候，类的代码和数据也不会被 packle 起来。只有 实例的数据被 packle 起来了。



## 如何 packle 类的实例（对象）

> 在多数情况下，不需要额外的代码就可以使类的实例 可被 pickle。
>
> By default, pickle will retrieve the class and the attributes of an instance via introspection.
>
> 当 unpickle 一个对象的实例的时候，`__init__` 方法是不会被调用的。
>
> The default behaviour first creates an uninitialized instance and then restores the saved attributes.
>
> The following code shows an implementation of this behaviour:

```python
def save(obj):
    return (obj.__class__, obj.__dict__)

def load(cls, attributes):
    obj = cls.__new__(cls)
    obj.__dict__.update(attributes) # 直接 update 了 dict。
    return obj
```



**getstate&setstate**

* `__getstate__` ，如果在类中实现了这个方法的话，pickle 的时候就会 pickle 这个方法返回的对象，而不是pickle  `__dict__` 中的东西。
* `__setstate__`，在 `unpickle` 的时候会被调用，如果实现了这个方法的话，被 packle 的对象就没必要是 dict， 但如果没有实现这个方法的话，那必须要是 dict 。

```python
class TextReader:
    """Print and number lines in a text file."""

    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename)
        self.lineno = 0

    def readline(self):
        self.lineno += 1
        line = self.file.readline()
        if not line:
            return None
        if line.endswith('\n'):
            line = line[:-1]
        return "%i: %s" % (self.lineno, line)

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['file']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)
        # Restore the previously opened file's state. To do so, we need to
        # reopen it and read from it until the line count is restored.
        file = open(self.filename)
        for _ in range(self.lineno):
            file.readline()
        # Finally, save the file.
        self.file = file
```

