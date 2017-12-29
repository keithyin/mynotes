# CMakeLists 教程



**你写了一些 source code （application）， 需要做什么？**

* 编译源代码
* 链接 其它 Libraries
* 通过 源代码的形式 或 二进制代码的形式 发行 application。



**如果可能的话，你还想做的事有**

* 执行 软件 测试
* 可再发行组件的 测试
* 观察运行结果



**编译源码**

```shell
# 手动方法
gcc -DMYDEFINE -c myapp.o myapp.cpp
```

手动方法具有以下缺点：

* you have many files
*  some files should be compiled only in a particular  platform, are you going to trust your brain?
* different defines depending on debug/release,  platform, compiler, etc

You'd like to automate this



**链接**

```shell
# 手动方法
ld -o myapp file1.o file2.o file3.o -lc -lmylib
```

* Again, unfeasiable if you have many files,  dependence on platforms, etc

You'd like to automate this.



**发行软件**

Traditional way of doing things:

*  Developers develop code
*  Once the software is finished, other people package it
*  There are many packaging formats depending on operating system version, platform, Linux distribution,  etc: .deb, .rpm, .msi, .dmg, .src.tar.gz,  .tar.gz, InstallShield, etc

You'd like to automate this but, is it possible to 
bring packagers into the development process?



**测试环节**

* You all use unit tests when you develop  software, don't you? You should!
* When and how to run unit tests? Usually a threestep process:
  * You manually invoke the build process (e.g. make)
  * When it's finished, you manually run a test suite
  * When it's finished, you look at the results and search for errors and/or warnings
  * Can you test the packaging? Do you need to invoke  the individual tests or the unit test manually?

**测试和收集结果**

* Someone needs to do testing for feach platform,  then merge the results
* Is it possible to automate this? “make test”? what  about gathering the results?





**什么是CMake**



## CMake 的编译系统

**编译器设置**

```shell
CMAKE_C_FLAGS  
CMAKE_CXX_FLAGS 
```



## 参考资料

[http://www.elpauer.org/stuff/learning_cmake.pdf](http://www.elpauer.org/stuff/learning_cmake.pdf)

[http://kfe.fjfi.cvut.cz/~holec/DATA/CMake-tutorial.pdf](http://kfe.fjfi.cvut.cz/~holec/DATA/CMake-tutorial.pdf)

[https://www.johnlamp.net/files/CMakeTutorial.pdf](https://www.johnlamp.net/files/CMakeTutorial.pdf)

[http://www.mi.fu-berlin.de/wiki/pub/ABI/PMSB_OpenMS_2011/Introduction_to_CMake.pdf](http://www.mi.fu-berlin.de/wiki/pub/ABI/PMSB_OpenMS_2011/Introduction_to_CMake.pdf)

[http://ilcsoft.desy.de/portal/e279/e346/infoboxContent560/CMake_Tutorial.pdf](http://ilcsoft.desy.de/portal/e279/e346/infoboxContent560/CMake_Tutorial.pdf)

