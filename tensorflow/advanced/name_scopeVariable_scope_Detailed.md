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

`Graph`中保存着一个属性`_name_stack`（string类型），`_name_stack`的值保存着当前的`name_scope`的名字，在这个图中创建的对象`Variable、Operation、Tensor`的名字之前都加上了这个前缀。