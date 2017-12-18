# pytorch学习笔记（十八）：扩展 pytorch-ffi

上篇博文已经介绍了如何通过 继承 `Function` ，然后使用`python` 来扩展 `pytorch`， 本文主要介绍如何通过 `cffi` 来扩展 `pytorch` 。



官网给出了一个 `MyAdd` 的 `Demo` [github地址](https://github.com/pytorch/extension-ffi)，本文通过 这个 `Demo` 来搞定如何 通过 `cffi` 来扩展 `pytorch`。



**从github上clone下来代码，目录结构是这样的**

* package： (这个部分的示例是可以将 创建的 扩展包安装到本机上)。 ???????????????
* script：（这个部分的示例 是 扩展包 仅当前 项目可见。）
* 在实际应用中，以上两个选择 而选一即可。



## 先看 script

cd 到这个 目录下，执行`python build.py` 来编译模块。 执行 `python test.py` 测试结果。 

* script
  * `functions`: (`op` 的 `Function` 封装)
    * `AddFunction.py`
  * `modules`:  (`op` 的 `Module` 封装)
    * `AddModule.py`
  * `src`: (存放 c，cuda 源码的地方)
    * `header.h`,  `source.c` 
  * `build.py`:  `python build.py` 编译模块
  * `test.py`:  `python test.py` 测试结果

