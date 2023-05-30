# 如何写 makefile与 CMakeLists

makefile 是一个组织代码编译的工具



**关于程序的编译与链接**

在此，我想多说关于程序编译的一些规范和方法，一般来说，无论是C、C++、还是pas，首先要把源文件编译成中间代码文件，在Windows下也就是 .obj 文件，UNIX下是 .o 文件，即 Object File，这个动作叫做编译（compile）。然后再把大量的Object File合成执行文件，这个动作叫作链接（link）。

链接时，主要是**链接函数和全局变量**，所以，我们可以使用这些中间目标文件（O文件或是OBJ文件）来链接我们的应用程序。链接器并不管函数所在的源文件，只管函数的中间目标文件（Object File），在大多数时候，由于源文件太多，编译生成的中间目标文件太多，而在链接时需要明显地指出中间目标文件名，这对于编译很不方便，所以，我们要给中间目标文件打个包，在Windows下这种包叫“库文件”（Library File)，也就是 .lib 文件，在UNIX下，是Archive File，也就是 .a 文件。

# Makefile

假设现在我们有三个文件，`hellomake.c`（主函数文件）， `hellofunc.c`（函数文件）， `hellomake.h`（头文件）。

```c++
// hellomake.c 主函数文件
#include <hellomake.h>

int main() {
  // call a function in another file
  myPrintHelloMake();

  return(0);
}
```



```c++
// hellofunc.c函数文件
#include <stdio.h>
#include <hellomake.h>

void myPrintHelloMake(void) {

  printf("Hello makefiles!\n");

  return;
}
```



```c++
// hellomake.h  头文件 
/*
example include file
*/

void myPrintHelloMake(void);
```



通常情况下： 我们会使用以下指令来编译代码：

```shell
gcc -o hellomake hellomake.c hellofunc.c -I.
```

* 将两个 `.c` 文件编译成 一个可执行文件 `hellomake`。
* `-I. `: 告诉 `gcc` 在找 头文件的时候要看一下当前文件夹。 
* 在不用 makefile 的情况下，在 `test/modify/debug` 循环中，我们要不停的执行这条 `gcc 指令`，但是一旦找不到这条 `gcc 指令`了，那就只能重新敲一遍了，（哭死。。）所以，makefile 还是必须要用的。



我们可以为上面的代码写一个 `makefile`：

```makefile
hellomake: hellomake.c hellofunc.c
     gcc -o hellomake hellomake.c hellofunc.c -I.
```

* 文件名字可以是 `Makefile` 或 `makefile`。
* 当写好这个文件后，然后在命令行 键入 `make` ，它就会执行在 `makefile`中定义好的编译命令。




**makefile文件的简单格式：**

```shell
<target>: [ <dependency > ]*
   [ <TAB> <command> <endl> ]+
```

* target : 通常是一个文件
* dependence:  target 依赖的文件
* 要运行的命令，基于 target 和  依赖




**makefile 介绍：**

----

make命令执行时，需要一个 Makefile 文件，以告诉make命令需要怎么样的去编译和链接程序。

首先，我们用一个示例来说明Makefile的书写规则。以便给大家一个感性认识。这个示例来源于GNU的make使用手册，在这个示例中，我们的工程有8个C文件，和3个头文件，我们要写一个Makefile来告诉make命令如何编译和链接这几个文件。我们的规则是：
​    1）如果这个工程没有编译过，那么我们的所有C文件都要编译并被链接。
​    2）如果这个工程的某几个C文件被修改，那么我们只编译被修改的C文件，并链接目标程序。
​    3）如果这个工程的头文件被改变了，那么我们需要编译引用了这几个头文件的C文件，并链接目标程序。

只要我们的Makefile写得够好，所有的这一切，我们只用一个make命令就可以完成，make命令会自动[智能](http://lib.csdn.net/base/aiplanning)地根据当前的文件修改的情况来确定哪些文件需要重编译，从而自己编译所需要的文件和链接目标程序。



**一 、makefile规则：**

在讲述这个Makefile之前，还是让我们先来粗略地看一看Makefile的规则。

```shell
 target ... : prerequisites ...
           command
            ...
            ...
```

   

​    target也就是一个目标文件，可以是Object File，也可以是执行文件。还可以是一个标签（Label），对于标签这种特性，在后续的“伪目标”章节中会有叙述。

​    prerequisites就是，要生成那个target所需要的文件或是目标。

​    command也就是make需要执行的命令。（任意的Shell命令）

这是一个文件的依赖关系，也就是说，target这一个或多个的目标文件依赖于prerequisites中的文件，其生成规则定义在command中。说白一点就是说，prerequisites中如果有一个以上的文件比target文件要新的话，command所定义的命令就会被执行。这就是Makefile的规则。也就是Makefile中最核心的内容。

说到底，Makefile的东西就是这样一点，好像我的这篇文档也该结束了。呵呵。还不尽然，这是Makefile的主线和核心，但要写好一个Makefile还不够，我会以后面一点一点地结合我的工作经验给你慢慢到来。内容还多着呢。：）



**二、一个示例**

正如前面所说的，如果一个工程有3个头文件，和8个C文件，我们为了完成前面所述的那三个规则，我们的Makefile应该是下面的这个样子的。

```makefile
    edit : main.o kbd.o command.o display.o /
          insert.o search.o files.o utils.o
            cc -o edit main.o kbd.o command.o display.o /
                       insert.o search.o files.o utils.o

    main.o : main.c defs.h
            cc -c main.c
    kbd.o : kbd.c defs.h command.h
            cc -c kbd.c
    command.o : command.c defs.h command.h
            cc -c command.c
    display.o : display.c defs.h buffer.h
            cc -c display.c
    insert.o : insert.c defs.h buffer.h
            cc -c insert.c
    search.o : search.c defs.h buffer.h
            cc -c search.c
    files.o : files.c defs.h buffer.h command.h
            cc -c files.c
    utils.o : utils.c defs.h
            cc -c utils.c
    clean :
            rm edit main.o kbd.o command.o display.o /
               insert.o search.o files.o utils.o
```

  

反斜杠（/）是换行符的意思。这样比较便于Makefile的易读。我们可以把这个内容保存在文件为“Makefile”或“makefile”的文件中，然后在该目录下直接输入命令“make”就可以生成执行文件edit。如果要删除执行文件和所有的中间目标文件，那么，只要简单地执行一下“make clean”就可以了。

在这个makefile中，目标文件（target）包含：执行文件edit和中间目标文件（*.o），依赖文件（prerequisites）就是冒号后面的那些 .c 文件和 .h文件。每一个 .o 文件都有一组依赖文件，而这些 .o 文件又是执行文件 edit 的依赖文件。依赖关系的实质上就是说明了目标文件是由哪些文件生成的，换言之，目标文件是哪些文件更新的。

在定义好依赖关系后，后续的那一行定义了如何生成目标文件的操作系统命令，一定要以一个`Tab`键作为开头。记住，make并不管命令是怎么工作的，他只管执行所定义的命令。

**make会比较targets文件和prerequisites文件的修改日期**，如果prerequisites文件的日期要比targets文件的日期要新，或者target不存在的话，那么，make就会执行后续定义的命令。

这里要说明一点的是，`clean` 不是一个文件，它只不过是一个动作名字，有点像[C语言](http://lib.csdn.net/base/c)中的lable一样，**其冒号后什么也没有，那么，make 就不会自动去找文件的依赖性，也就不会自动执行其后所定义的命令**。要执行其后的命令，就要在make命令后明显得指出这个lable的名字。这样的方法非常有用，我们可以在一个makefile中定义不用的编译或是和编译无关的命令，比如程序的打包，程序的备份，等等。



### 定义常量

```makefile
CC=gcc
CFLAGS=-I.

hellomake: hellomake.o hellofunc.o
     $(CC) -o hellomake hellomake.o hellofunc.o $(CFLAGS)

```

* `CFALGS` : 传到编译命令的的一些 `flags` 
* 这里是不需要写 编译 `.o` 的代码的, `make` 会自己推断, 但是这样会存在一个问题, 如果头文件变了, `make` 并不会重新编译代码. `(因为, 并没有显示指明 .o文件的依赖)`

### 啥玩意?

```makefile
CC=gcc
CFLAGS=-I.
DEPS = hellomake.h

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

hellomake: hellomake.o hellofunc.o 
	$(CC) -o hellomake hellomake.o hellofunc.o
```

* 有点像 `shell` 的
* `%` : 应该是个类似通配符的东西, `.c` 文件去哪里找啊, `make` 会自动脑补
* `$(CC)`
  * `-c` : 编译成 目标文件 `.o`
  * `-o $@` : 将编译结果的输出名字设置为 `:` 左边的值
  * `$<` : 依赖列表的第一项



### 啥玩意2

```makefile
CC=gcc
CFLAGS=-I.
DEPS = hellomake.h
OBJ = hellomake.o hellofunc.o 

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

hellomake: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)
```

* `$^`  :  `:` 右边的值
* `$@` :  `:` 左边的值



### 啥玩意3

> `.h` 放在一个文件夹内, `.c` 放在一个文件夹内, `.lib` 放在一个文件夹内

```makefile
IDIR =../include
CC=gcc
CFLAGS=-I$(IDIR)

ODIR=obj
LDIR =../lib

LIBS=-lm

_DEPS = hellomake.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = hellomake.o hellofunc.o 
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

hellomake: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 
```

* The `.PHONY` rule keeps `make` from doing something with a file named clean.
* 



# CMakeLists

**CMake:** 用来生成 Makefile。

* 编译流水线
  * `CmakeLists.txt` 
  * `Makefile`
  * `.obj, .o`
  * `.exe, .dll, .lib, .a, .o, .dylib`

## cmake 工作流

```
myapp
	build
	trunk
		CmakeLists.txt
```

* 拥有上面一个目录树

```shell
cd myapp/build
cmake ../trunk
make
```

* 如何展示更多的`cmake` build 信息

```shell
SET( CMAKE_VERBOSE_MAKEFILE on )
# 或者
$ make VERBOSE=1

$ export VERBOSE=1
$ make
```



**添加其它source**

<img src="../imgs/cmake-1.png" style="zoom:30%;" />

**改变编译参数**

* `cmake` 通常使用默认的 预处理器, 编译器, 和链接器
* 如何修改 `preprocessor` : ADD_DEFINITIONS and REMOVE_DEFINITIONS
* 编译器设置:  CMAKE_C_FLAGS and CMAKE_CXX_FLAGS variables



**控制流**

```cmake
if(expression)
...
else(expression)
...
endif(expression)

# 处理list
foreach(loop_var)
...
endforeach(loop_var)

while(condition)
...
endwhile(condition)
```



**管理 debug 和 release build**

* `SET(CMAKE_BUILD_TYPE Debug)`
* As any other variable, it can be set from the command line: cmake ­DCMAKE_BUILD_TYPE=Release ../trunk
* Specify debug and release targets and 3rdparty libs:  (<font color='red'>并不知道说的啥</font>)
  * TARGET_LINK_LIBRARIES(wakeup RELEASE \${wakeup_SRCS}) 
  * TARGET_LINK_LIBRARIES(wakeupd DEBUG $ {wakeup_SRCS})



**需要依赖其它的库**

* 如果第三方库的 `.h` 和 其库在 `PATH/LD_LIBRARY_PATH` (标准文件夹) 中, 在 `CmakeLists.txt` 无需其它操作
* 如果没有在标准文件夹中
  * 头文件: 使用 `INCLUDE_DIRECTORIES`
  * libraries: 使用 `find_library` , 然后链接其结果

**查找已经安装的软件**

* `FIND_PACKAGE( Qt4 REQUIRED )`
* Cmake includes finders (FindXXXX.cmake) for ~130 software packages, many more available in Internet
*  If using a non­CMake FindXXXX.cmake, tell Cmake where to find it by setting the CMAKE_MODULE_PATH variable
*  Think of FIND_PACKAGE as an #include

## 基本概念

* `CmakeLists.txt` : 
  * Input text files that contain the project parameters and describe the flow control of the build process in simple CMake language.
* `CMakeModules`
  * Special cmake file written for the purpose of finding a certain piece of software and to set it's libraries, include files and definitions into appropriate variables so that they can be used in the build process of another project. (e.g. FindJava.cmake, FindZLIB.cmake, FindQt4.cmake)

**两个tree**

* 关注的问题: Where to place object files, executables and libraries?
* `Source tree` 包含, **这个是我们所写的代码的 目录树**
  * cmake input files (`(CmakeLists.txt`)
  * 代码源文件 `.cpp` , 代码头文件 `.h`
* `Binary tree` 包含, **这个是编译后的结果所在的 目录树**
  * Native build system files (Makefiles)
  * Output from build process:
    * Libraries 
    *  Executables 
    * Any other build generated file
* 如果是 `in-source` 编译, `source-tree` 和 `binary-tree` 在同一个文件夹下; 如果是 `out-source` 编译, 他俩在不同的文件夹下.  **目前所见的都是采用的 out-source 编译方式**

**几个宏**

* `CMAKE_MODULE_PATH` : Path to where the CMake modules are located
* `CMAKE_INSTALL_PREFIX`: Where to put files when calling 'make install'
* `CMAKE_BUILD_TYPE` : Type of build (Debug, Release, ...)
* `BUILD_SHARED_LIBS`: Switch between shared and static libraries
* 这些值可以在`CmakeList.txt` 文件中指定, 也可以在命令行中指定
  * `by prefixing a variable's name with '-D':`  `cmake -DBUILD_SHARED_LIBS=OFF`

**如何使用 cmake (out-of-source) 编译**

* 构建编译文件夹 : `mkdir build & cd build` 
* 配置: `cmake [options] <source-tree>`
* 编译: `make`
* 安装 : `make install`
* 后两个可以整合成一个操作, 只用最后一个就行了.

**基础语法**

* `add_library(target_name source_file)` : 
  * 编译成一个静态链接库, 下面 `Target & Object` 中解释道, 这命令实际是构建一个对象.
  * creates an static library from the listed sources
  * `add_library(target_name SHARED source_files)` : 
    * 添加`Shared` 表明了是创建一个动态链接库. 下面 `Target & Object` 中解释道, 这命令是在给对象添加属性.
  * 静态链接库  与 动态链接库
    * 静态链接库:  on linking, add the used code to your executable
    * 动态链接库:  on linking, tell the executable where to find some code it needs
  * `cmake -D BUILD_SHARED_LIBS=TRUE` 这个会设置 `add_library` 的默认为 `SHARED`



* `add_executable(target_name source_file)` : 编译成一个可执行文件
* `target_link_libraries(executable_name some_lib)` : 编译链接指示, 将其它的库链接到当前的 target 中
  * 如果 `some_lib` 是静态库的话, 就是静态链接, 如果 `some_lib` 是动态库的话, 就动态链接.
  * 需要链接静态库的时候使用
  * 对每个目标设置链接
* `link_libraries(lib1 lib2)` : 所有的 `target` 都链接同一组 `libs`
* 命令 : `command(arg1 arg2 ...)`
* 列表: `A;B;C` ,使用分号分隔
* 变量: `${var}`
  * 不需要声明, 通常情况下不需要指定类型
  * `set(var value)` : 负责创建和修改变量的值
  * SET can do everything but LIST makes some operations easier
  * Use `SEPARATE_ARGUMENTS` to split space-separated arguments (i.e. a string) into a list (semicolon­-separated)
* `include_directories(dir1 di2)` : 设置可查找的头文件目录.
* `aux_source_directories(source)` :
* `add_custom_target`
* `add_denpendices(target1 dep1 dep2)` : target1 依赖于 dep1, dep2
* `add_definitions(-Wall -ansi)`



**工具方法**

* `set(var value)` : 设置变量的值
* `find_file`
* `find_library()`
* `find_program()`
* `find_package()`
* `message(STATUS ....)` : 控制台打印一些信息

## CmakiList.txt中可以使用的工具变量

```cmake
${CMAKE_CURRENT_SRC_DIR} # cmake文件所在的目录
```



## Target 与 Object

* cmake 中 有 `target` 的概念, 这个概念源自 `makefile`, 在 `cmake` 中, 我们可以将 `target` 看做 `object(面向对象编程中的对象)` .

```cmake
# 构造器: constructors
add_executable()
add_library()

# 一堆成员变量: target properties

# 成员函数, 用来重写 target 的属性值的
get_target_property()
set_target_properties()
get_property(TARGET)
set_property(TARGET)

# 设置一些宏, 可以在代码中使用的这种. 可以 PUBLIC, PRIVATE, INTERFACE
target_compile_definitions()
target_compile_features()
target_compile_options()
# 设置 target 的 include_directories, 方便其它的 target 使用 该 target
target_include_directories()
target_link_libraries()
target_sources()
```

## PUBLIC PRIVATE INTERFACE

```cmake
target_include_directories(some_target PUBLIC ${CMAKE_CURRENT_SRC_DIR}/src)

# 这样的话, some_target 的代码中是可以使用  VERSION 得到 3 这个值
target_compile_definitions(some_target PUBLIC VERSION=3)
```

* `PUBLIC` : 含义是, 如果其它 `target` 链接了这个 `target` 的话, 也会自动设置这个 `include dir`
* `PRIVATE`: 含义是, 这个只是当前 `target` 独享, 并不会 `populate` 给其它 `target`
* `INTERFACE`: 含义是, 自己当前 不用,  但是会 `populate` 给其它 `link` 这个 `target` 的

## 最简单一版

```cmake
cmake_minimum_required(VERSION 3.10)

# set the project name, 这个命令并非强制使用, 但是最好还是用着
# 设置的名字可以通过 ${PROJECT_NAME} 取得, version 可以通过 ${PROJECT_VERSION} 取得
project(Tutorial VERSION 1.0.0)

# add the executable
add_executable(Tutorial tutorial.cxx)
```



```shell
set(VarName value) # 用于设置变量值

add_executable(exe_name source_files) # 创建可执行文件

add_library

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)
```

[https://cmake.org/cmake/help/v3.10/command/target_link_libraries.html?highlight=target_link_#command:target_link_libraries](https://cmake.org/cmake/help/v3.10/command/target_link_libraries.html?highlight=target_link_#command:target_link_libraries)



* library :
* package : 



**step1**

> 源码编译可执行文件

```shell
cmake_minimum_required (VERSION 2.6)
project (Tutorial)
add_executable(Tutorial tutorial.cxx)
```

**step2**

> 添加版本号 和 用于配置的头文件

```shell
cmake_minimum_required (VERSION 2.6)
project (Tutorial)
# The version number. set， 设置变量和值。
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)

# 配置一个头文件，将 CmakeList 中的一些配置 植入到 源码中
# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/TutorialConfig.h.in"
  "${PROJECT_BINARY_DIR}/TutorialConfig.h"
  )
 
# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")
 
# add the executable
add_executable(Tutorial tutorial.cxx)
```





```shell
project(AtenDemo1)

# set , 设置变量名对应的值。之后可以用 ${param_name} 来获取值
set(CMAKE_CXX_STANDARD 11) 
set(EXT_DIR /home/keith/Programs/aten_install/)

find_library(LIB libATen.so ${EXT_DIR}/lib)

include_directories(${EXT_DIR}/include)

link_libraries(${LIB})

set(SOURCE_FILES main.cpp maskrcnn/maskrcnn.h maskrcnn/maskrcnn.c.cpp roi/roi.h roi_align/roi_align.h roi_align/cuda/roi_align_kernel.h)
add_executable(AtenDemo1 ${SOURCE_FILES})
```



## 参考资料

[http://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/](http://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/)

[https://www.cs.umd.edu/class/fall2002/cmsc214/Tutorial/makefile.html](https://www.cs.umd.edu/class/fall2002/cmsc214/Tutorial/makefile.html) 

[https://cmake.org/cmake-tutorial/](https://cmake.org/cmake-tutorial/)
[https://cmake.org/cmake/help/latest/guide/tutorial/A%20Basic%20Starting%20Point.html](https://cmake.org/cmake/help/latest/guide/tutorial/A%20Basic%20Starting%20Point.html)

[http://www.elpauer.org/stuff/learning_cmake.pdf](http://www.elpauer.org/stuff/learning_cmake.pdf)
