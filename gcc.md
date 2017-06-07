# 闲谈

## 环境变量

* `windows` 中到的 `path` 与 `linux`中的 `PATH` (windows对大小写不敏感，linux对大小写敏感)
  * 系统的环境变量：当系统的接口接收到一个程序启动命令时，除了在挡墙的目录下找那个可执行的文件以外，还可能需要到`PATH(linux), path(windows)`指定的路径寻找。

## GCC 程序环境变量

### include 搜索路径

* 当我们`#include "headfile.h"` 时，搜索顺序为：
  * 当前目录
  * 搜索 `-I` 参数指定的目录
  * 搜索 `g++` 环境变量`CPLUS_INCLUDE_PATH`（`C程序使用的是C_INCLUDE_PATH`）
  * 最后搜索 `g++` 的内定目录 (`/usr/include,/usr/local/include,/usr/lib/gcc/x86_64-redhat-linux/4.1.1/include`)
  * 找到了就不再找了。


* 当我们`#include <headfile.h>` 时，搜索顺序为：
  * 搜索 `-I` 参数指定的目录
  * 搜索 `g++` 环境变量`CPLUS_INCLUDE_PATH`（`C程序使用的是C_INCLUDE_PATH`）
  * 最后搜索 `g++` 的内定目录 (`/usr/include,/usr/local/include,/usr/lib/gcc/x86_64-redhat-linux/4.1.1/include`)
  * 找到了就不再找了。



### 库搜索路径

[此部分来自http://www.cnblogs.com/panfeng412/archive/2011/10/20/library_path-and-ld_library_path.html ](http://www.cnblogs.com/panfeng412/archive/2011/10/20/library_path-and-ld_library_path.html)

**改正了其错误地方**

这是是用在链接的时候的。这时候涉及到的环境变量：

* LIBRARY_PATH

这是是用在程序运行的时候的。这时候涉及到的环境变量：

* LD_LIBRARY_PATH

### LIBRARY_PATH

LIBRARY_PATH环境变量用于在**程序链接期间**查找**动态/静态链接库时**指定查找共享库的路径，例如，指定g++编译需要用到的动态/静态链接库的目录。设置方法如下（其中，LIBDIR1和LIBDIR2为两个库目录）：

```shell
export LIBRARY_PATH=LIBDIR1:LIBDIR2:$LIBRARY_PATH
```



## LD_LIBRARY_PATH

LD_LIBRARY_PATH环境变量用于在**程序加载运行期间**查找**动态链接库时**指定除了系统默认路径之外的其他路径，注意，LD_LIBRARY_PATH中指定的路径会在系统默认路径之前进行查找。设置方法如下（其中，LIBDIR1和LIBDIR2为两个库目录）：

```shell
export LD_LIBRARY_PATH=LIBDIR1:LIBDIR2:$LD_LIBRARY_PATH
```



### 区别与使用

开发时，设置LIBRARY_PATH，以便gcc能够找到**链接时**需要的动态/静态链接库。

发布时，设置LD_LIBRARY_PATH，以便程序**加载运行时**能够自动找到需要的动态链接库。



As pointed below, your libraries can be static or shared. If it is static then the code is copied over into your program and you don't need to search for the library after your program is compiled and linked. If your library is shared then it needs to be dynamically linked to your program and that's when `LD_LIBRARY_PATH` comes into play.





## 动态链接库与静态链接库

* windows 上
  * 静态链接库(.lib)
  * 动态链接库 (.dll)
* linux 上
  - 静态链接库(.a)
  - 动态链接库 (.so)
  - linux下静态链接库命名规范 "lib[you_library_name].a"

## 可能碰到的问题

* `-I`指令指定的 包含路径 使用`#include <header>` 的方式可能依旧找不到，可以尝试 `#include "header.hpp"` 



## Netbeans开发caffe

* 创建好项目
* 右击项目名字
* 左击属性
* `Build` --> `c++ complier` --> `include directories` (把caffe的路径填进去 "caffe/include")，把CUDA的include也放进去
* Linker --> libraries (把libraries指定好！！)
  * `add library file` --> `libcaffe.a` 或者 `libcaffe.so`的绝对路径
  * `add library`  --> `boost_system`




## 设置动态链接库路径方法

* 环境变量LD_LIBRARY_PATH指定的动态库搜索路径(修改/etc/profile 文件中的 `LD_LIBRARY_PATH`)
* 在 `/etc/ld.so.conf/` 中新添加一个文件 `your.conf`
  * 然后在文件中添加动态库路径
* 默认的动态库搜索路径`/lib, /usr/lib `




## 问题总结

* cannot find -lcaffe

## 参考资料

[http://www.cnblogs.com/panfeng412/archive/2011/10/20/library_path-and-ld_library_path.html](http://www.cnblogs.com/panfeng412/archive/2011/10/20/library_path-and-ld_library_path.html)

[https://stackoverflow.com/questions/4250624/ld-library-path-vs-library-path](https://stackoverflow.com/questions/4250624/ld-library-path-vs-library-path)