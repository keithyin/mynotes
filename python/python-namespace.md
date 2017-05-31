# python的命名空间

`python`使用命名空间记录变量。`python`中的命名空间就像是一个`dict`，`key`是变量的名字，`value`是变量的值。

* `python`中，每个函数都有一个自己的命名空间，叫做`local namespace`，它记录了函数的变量。
* `python`中，每个`module`有一个自己的命名空间，叫做`global namespace`，它记录了`module`的变量，包括 `functions, classes` 和其它` imported modules`，还有 `module`级别的 变量和常量。
* 还有一个`build-in` 命名空间，可以被任意模块访问，这个`build-in`命名空间中包含了`build-in function` 和 `exceptions`。



当`python`中的某段代码要访问一个变量`x`时，`python`会在所有的命名空间中寻找这个变量，查找的顺序为:

* local namespace - 指的是当前函数或者当前类方法。如果在当前函数中找到了变量，停止搜索
* global namespace - 指的是当前的模块。如果在当前模块中找到了变量，停止搜索
* build-in namespace - 如果在之前两个`namespace`中都找不到变量`x`，`python`会假设`x`是`build-in`的函数或者变量。如果`x`不是内置函数或者变量，`python`会报错`NameError`。



*对于闭包来说，这里有一点区别，如果在local namespace中找不到变量的话，还会去父函数的local namespace中找变量。*



## locals

内置函数`locals()`, 返回当前函数（方法）的局部命名空间

```pyth
def func(a = 1):
    b = 2
    print(locals())
    return a+b
func()
# {'a': 1, 'b': 2} 可以看出，locals返回的是个dict
```



## globals

内置函数`globals()`，返回当前`module`的命名空间

```pyt
def func(a = 1):
    b = 2
    return a+b
func()
print(globals()) # globals()返回的也是个dict
```



**locals()和globals()有一个区别是，locals只读，globals可以写**



```python
def func(a = 1):
    b = 2
    return a+b
func()
glos = globals()
glos['new_variable'] = 3
print(new_variable)
# 3  , 我们并没有显示定义new_variable这个变量，只是在globals中添加了这个key，在随后的代码中，
#就可以像访问一般变量一样来访问。

def func(a = 1):
    b = 2
    locs = locals()
    locs['c']  = 1
    print(c)
func()
# NameError: name 'c' is not defined
```



## from module import 和 import module

* 使用`import module`时，`module`本身被引入，但是保存它原有的命名空间，所以我们需要使用`module.name`这种方式访问它的 函数和变量。
* `from module import`这种方式，是将其它模块的函数或者变量引到当前的命名空间中，所以就不需要使用`module.name`这种方式访问其它的模块的方法了。



## if \_\_name\_\_ trick

`python`中的`module`也是对象，所有的`modules`都有一个内置的属性`__name__`，模块的`__name__`属性的值取决于如何使用这个模块，如果`import module`，那么`__name__`属性的值是模块的名字。如果直接执行这个模块的话，那么`__name__`属性的值就是默认值`__main__`。

