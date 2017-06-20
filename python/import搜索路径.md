# python import 搜索路径

当我们 `import module` 的时候，python是怎么找到我们想引入的模块的？

## 1. 找 module cache
使用 `import` 的时候， 第一个搜索的地方就是 `sys.modules`。 它是一个 `dict` ， 保存了所有之前被 `import` 的 `module`。如果我们之前 执行了以下语句
```python
import foo.bar.baz

```
那么 `foo,   foo.bar` 都会在 `sys.modules` 里面。

##  2. 使用finders 和 loaders

* finders : 找模块
* loaders: 加载模块

如果 `sys.modules` 中没有要引入的 `module`的时候，`python` 就要用 `finders` 来帮忙找模块了。

`python` 有三种默认的 `finders`：

* 知道如何 定位 `build-in` 模块的 `finder`
* 知道如何 定位 `frozen module` 的 `finder`
* 知道如何查找 `import path` 中的模块的 `finder`

**import path：**
一个 `location` 列表，`finder` 去里面找想要引入的模块。这个列表通常来自于 `sys.path`。 `sys.path`  中的路径是可以动态加载的。

## python 如何设置 sys.path 的

* OS paths that have your system libraries
* current directory python started from
* environmental variable $PYTHONPATH
* you can add paths at runtime.

## module 和 package

* module : 一个 `.py` 文件。
* package : 包含 `.py` 文件的目录 ，里面包含一个 `__init__.py` 来告诉 python，这是一个 package， 你可以从这里引入module。

## 参考资料
[https://askubuntu.com/questions/470982/how-to-add-a-python-module-to-syspath](https://askubuntu.com/questions/470982/how-to-add-a-python-module-to-syspath)
