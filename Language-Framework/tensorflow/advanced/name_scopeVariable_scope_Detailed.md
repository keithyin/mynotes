# tensorflow的上下文管理器，详解namescope和variablescope

## with block 与上下文管理器

* 上下文管理器：意思就是，在这个管理器下做的事情，会被这个管理器管着。

  熟悉一点python的人都知道，with block与上下文管理器有着不可分割的关系。为什么呢？因为`with Object() as obj:`的时候，会自动调用`obj`对象的`__enter__()`方法，而当出去`with block`的时候，又会调用`obj`对象的`__exit__`方法。正是利用 `__enter__()和__exit__()`，才实现类似上下文管理器的作用。

* `tensorflow`中的`tf.name_scope`和 `variable_scope`也是个作为上下文管理器的角色



## variable_scope

* `tensorflow`怎么实现`variable_scope`上下文管理器这个机制呢？

要理解这个，首先要明确`tensorflow`中，`Graph`是一个就像一个大容器，`OP、Tensor、Variable`是这个大容器的组成部件。

* `Graph`中维护一个`collection`，这个`collection`中的 键`_VARSCOPE_KEY`对应一个 `[current_variable_scope_obj]`，保存着当前的`variable_scope`。


## name_scope

`Graph`中保存着一个属性`_name_stack`（string类型），`_name_stack`的值保存着当前的`name_scope`的名字，在这个图中创建的对象`Variable、Operation、Tensor`的名字之前都加上了这个前缀


## 源码理解

**variable_scope**
首先看`tf.variable_scope(..)` 我们会看到以下代码，之前介绍说：`Graph`中维护一个`collection`，这个`collection`中的 键`_VARSCOPE_KEY`对应一个 `[current_variable_scope_obj]`，保存着当前的`variable_scope`。 所以我们通过源码来看一下。

* 什么时候 tensorflow 将当前的 variable_scope 放到collection中。
* 当从一个 variable_scope 出来的时候，tensorflow 是如何将之前的 variable_scope 放到 collection中

追下源码，首先看到的是：
```python
@tf_contextlib.contextmanager
def variable_scope(name_or_scope,
                   default_name=None,
                   values=None,
                   initializer=None,
                   regularizer=None,
                   caching_device=None,
                   partitioner=None,
                   custom_getter=None,
                   reuse=None,
                   dtype=None,
                   use_resource=None):
```
首先看到，有一个`decorator`，对于python的 `decorator` 可以简单的理解成函数的一个`wrapper` [详见](https://github.com/KeithYin/mynotes/blob/master/python/decorator.md)。 先不管 `decorator`， 来看看 `variable_scope`中有什么有趣的东西。

```python

  with _pure_variable_scope(
      name_or_scope,
      reuse=reuse,
      initializer=initializer,
      regularizer=regularizer,
      caching_device=caching_device,
      partitioner=partitioner,
      custom_getter=custom_getter,
      old_name_scope=old_name_scope,
      dtype=dtype,
      use_resource=use_resource) as vs:
    yield vs
```
`tf.variable_scope()` 都是用已这种方式终结的。然后我们来看一下， `_pure_variable_scope()`， 它里面做了什么神奇的操作。

```python
@tf_contextlib.contextmanager
def _pure_variable_scope(name_or_scope,
                         reuse=None,
                         initializer=None,
                         regularizer=None,
                         caching_device=None,
                         partitioner=None,
                         custom_getter=None,
                         old_name_scope=None,
                         dtype=dtypes.float32,
                         use_resource=None):
```
继续不考虑 `decorator`， 直接看 `_pure_variable_scope()` 代码，就会发现以下代码片段：

```python
#这句取得是，之前一个 varScope 对象
default_varscope = ops.get_collection_ref(_VARSCOPE_KEY)

# 把之前的varScope对象 保存起来
old = default_varscope[0]


# 这部分可以看到，new_name 的生成，是 old_name+"/"+name_or_scope 
# 这就说明了，为什么varScope的名字是层层嵌套的
if isinstance(name_or_scope, VariableScope):
    new_name = name_or_scope.name
else:
    new_name = old.name + "/" + name_or_scope if old.name else name_or_scope

# 这里是 创建了一个新的 VariableScope对象，并将他放入到collection中
# 名字是 new_name
default_varscope[0] = VariableScope(
          reuse,
          name=new_name,
          initializer=old.initializer,
          regularizer=old.regularizer,
          caching_device=old.caching_device,
          partitioner=old.partitioner,
          dtype=old.dtype,
          use_resource=old.use_resource,
          custom_getter=old.custom_getter,
          name_scope=old_name_scope or name_or_scope)

# 这句是将当前的 varScope yield 出来，
yield default_varscope[0]
# 到现在为止，我们已经看到tensorflow 是如何将新创建的varScope放入到
# collection中， 那么，什么地方将旧的 varScope 重新放回 collection 中呢？

# 再继续看部分的代码
var_store.close_variable_subscopes(new_name)
    # If jumping out from a non-prolonged scope, restore counts.
if isinstance(name_or_scope, VariableScope):
      var_store.variable_scopes_count = old_subscopes
default_varscope[0] = old

# 在yield 语句不远处，我们看到了 将旧的 varScope 重新放回 collection中
# 的代码

```
要想更清楚的理解这一部分，时候看一波 `decorator` 的实现了。

```python
def contextmanager(target):
# target 就是被 decorator 的 函数啦
# 继续追一下，发现context_manager 是一个
#_GeneratorContextManager对象
# 继续看一下这个对象的细节
  context_manager = _contextlib.contextmanager(target)
  
  return tf_decorator.make_decorator(target, context_manager, 'contextmanager')
```


```python

class _GeneratorContextManager(ContextDecorator):
    """Helper for @contextmanager decorator."""
    def __init__(self, func, args, kwds):
        self.gen = func(*args, **kwds) #没有被decorator的函数
        self.func, self.args, self.kwds = func, args, kwds
        doc = getattr(func, "__doc__", None)
        if doc is None:
            doc = type(self).__doc__
        self.__doc__ = doc

    def __enter__(self):
        try:
            return next(self.gen) #执行函数到yield 语句
        except StopIteration:
            raise RuntimeError("generator didn't yield") from None

    def __exit__(self, type, value, traceback):
        if type is None:
            try:
                next(self.gen) # 执行函数yield 语句后面的部分
            except StopIteration:
                return
            else:
                raise RuntimeError("generator didn't stop")

```


```python

with _pure_variable_scope(
      name_or_scope,
      reuse=reuse,
      initializer=initializer,
      regularizer=regularizer,
      caching_device=caching_device,
      partitioner=partitioner,
      custom_getter=custom_getter,
      old_name_scope=old_name_scope,
      dtype=dtype,
      use_resource=use_resource) as vs:
      
#进入with block的时候，创建了一个_GeneratorContextManager对象
# 执行这个对象的 __enter__ 方法， 没有加decorator的
# _pure_variable_scope 函数执行到
# yield default_varscope[0] 部分， yield出来的varScope被返回。

# 当退出当前with block的时候， _GeneratorContextManager对象执行
# __exit__() 方法 执行 没有加decorator的_pure_variable_scope 函数的
# 剩余部分，这时，旧的 varScope 被重新载入

```

**name_scope**
`name_scope` 和 `variable_scope` 的实现形式差不多，都涉及到了 `@tf_contextlib.contextmanager` 和 `_GeneratorContextManager`
可以看出，就是因为 `with block` 需要 一个有 `__enter__ , __exit__` 方法的对象，所以才搞出来这个一个类。

核心的地方在这
```python
if name:
     if self._name_stack:
        if not _VALID_SCOPE_NAME_REGEX.match(name):
          raise ValueError("'%s' is not a valid scope name" % name)
     else:
        # Scopes created in the root must match the more restrictive
        # op name regex, which constrains the initial character.
        if not _VALID_OP_NAME_REGEX.match(name):
          raise ValueError("'%s' is not a valid scope name" % name)
try:
     old_stack = self._name_stack
     if not name:  # Both for name=None and name="" we re-set to empty scope.
        new_stack = None
     elif name and name[-1] == "/":
        new_stack = _name_from_scope_name(name)
     else:
        new_stack = self.unique_name(name)
      self._name_stack = new_stack
      # yield 新 name scope 的地方
      yield "" if new_stack is None else new_stack + "/"
    finally:
    # 加载旧的 name_space 的地方
      self._name_stack = old_stack
  # pylint: enable=g-doc-return-or-yield

```

**slim.arg_scope()** 就不是用以上的策略存储新旧 scope 了， 它是直接保存在一个栈中的。

```python
try:
     _get_arg_stack().append(current_scope)
     yield current_scope
finally:
     _get_arg_stack().pop()
```
