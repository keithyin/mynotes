# 1 基础

## 最基础版本
**最上层** CmakeLists.txt需要包含
```
# 所需的cmake版本, 非上层的CMakeLists.txt可以不用指定cmake版本
cmake_minimum_required(VERSION 3.15.0)

# 设置项目名 和 版本号。该行紧跟 cmake_minimum_required 后面
project(demo1 VERSION 0.1.0)

# create an executable using the specified source code files.
add_executable(executable_name source_file)
```

写了之后如何使用
```
# 1. 创建一个 build 文件夹
mkdir build_dir & cd build_dir
# 2. 配置
cmake ..
# 3. 构建
cmake --build .
```

## 设置C++版本

cmake有些特殊变量，这些变量要么是后台创建的，或者是对cmake有意义的，这些变量基本都以`CMAKE_`打头

```
# CMAKE_CXX_STANDARD 设置 C++语言标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
```

## 源码中读取cmake中配置的版本号
> cmake中配置版本号，C++代码中可以读到。通过configure_file()实现. 基本思路是 准备一个模板头文件，里面留一些 占位符，然后通过 configure_file，将里面的占位符 用 cmake变量替代并生成一个新的头文件，源码中只需要 include 生成的头文件就可以了

1. 通过project(demo1 VERSION 0.1.0) 设置版本号, project之后，cmake就在后台定义了 <PROJECT-NAME>_VERSION_MAJOR，<PROJECT-NAME>_VERSION_MINOR
```
project(demo1 VERSION 0.1.0)
```

2. 使用 configure_file(). 将配置文件中的变量值由cmake的变量替换. 替换后生成的文件在build文件夹中。

```
configure_file(Config.h.in Config.h)
```

3. 由于代码中需要依赖config文件，所以需要使用  target_include_directories 
```
target_include_directories(demo1 PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )
```

4. 当调用configure_file时。config文件中的 @SOME_NAME@ 会被cmake变量替代。然后在build文件夹下生成一个 Config.h的文件
```c
// the configured options and settings for Tutorial
#define VERSION_MAJOR @demo1_VERSION_MAJOR@
#define VERSION_MINOR @demo1_VERSION_MINOR@
```

5. 代码中 include Config.h 文件即可
   
```c
#include "Config.h"

if (argc < 2) {
// report version
std::cout << argv[0] << " Version " << Tutorial_VERSION_MAJOR << "."
            << Tutorial_VERSION_MINOR << std::endl;
std::cout << "Usage: " << argv[0] << " number" << std::endl;
return 1;
}
```

6. 最终的CMakeLists.txt如下
```cmake
cmake_minimum_required(VERSION 3.0.0)

project(demo1 VERSION 0.1.0)

configure_file(Config.h.in Config.h)

add_executable(demo1 demo1.cpp)

target_include_directories(demo1 PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )
```

# 2. 创建 & 使用 library
> 将大项目拆成多个小目录，每个目录都是一个 library. 
> 基本思路是 最上层 CMakeLists.txt 通过  add_subdirectory() 说明要构建子目录。因为要构建子目录，所以子目录也需要有个 CMakeLists.txt说明自己应该如何构建。 子目录构建好了之后，最上层的CMakeLists.txt需要 引入对应的目录 和 构建的 lib 才能构建成最终的大项目

1. 创建一个新子目录，里面加一个 CMakeLists.txt, 里面包含 add_library
2. 最上层 CMakeLists.txt中 add_subdirectory()
3. 构建最终的 target的时候 还要  target_include_directories()头文件,target_link_libraries() 链接lib

```cmake
# 最上层
cmake_minimum_required(VERSION 3.0.0)

project(demo1 VERSION 0.1.0)

add_subdirectory(sub_dir_name)

add_executable(demo1 demo1.cpp)

target_link_libraries(demo1 PUBLIC lib_name)
target_include_directories(demo1 PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/sub_dir_name"
                          )


```

```cmake
# 子目录下的
add_library(lib_name source_files)
```

## optional
> 用来做条件编译，option() 会提供一个变量，这个变量可以在配置cmake编译的时候设置。该值会被缓存。所以当在一个build目录执行cmake时，没必要每次都设置这个值。
> cmake中根据option的值，选择是否执行某些逻辑。源码中，根据config.h.in定义的宏来执行条件编译

```cmake
# 配置cmake选项 -D后面的就是option定义的变量
cmake .. -DUSE_MATH_TOOLS=OFF
# 编译
cmake --build .
```

```cmake
# top level cmake

option(USE_MYMATH "Use tutorial provided math implementation" ON)

configure_file(Config.h.in Config.h)

if(USE_MYMATH)
  add_subdirectory(MathFunctions)
  list(APPEND EXTRA_LIBS MathFunctions)
  list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/MathFunctions")
endif()

# 如果 USE_MYMATH=ON。那么就会链接。否则，不会链接。注意代码中也要处理这种情况
target_link_libraries(Tutorial PUBLIC ${EXTRA_LIBS})
target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           ${EXTRA_INCLUDES}
                           )
```

```c++
// Config.h.in
#cmakedefine USE_MYMATH
```

```c++
// main.cpp
#include "Config.h"

#ifdef USE_MYMATH
    #include "my_math.h"
#else

#endif

#ifdef USE_MYMATH
  const double outputValue = mysqrt(inputValue);
#else
  const double outputValue = sqrt(inputValue);
#endif
```

# 3. Usage Requirements for a Library
> library自己定义自己的 使用所需条件。一旦定义了，该使用条件就会传递给调用方。具体的例子为，在上面的demo中，上层target依赖子目录taget的时候，需要 target_include_directories 和 target_link_libraries。如果子目录定义了使用自己时所需的头文件，那么 上层CMakeLists.txt中就不用 target_include_directories 了。


```cmake
# subdirectory CMakeLists.txt
add_library(my_lib lib.cpp)

# 加上下面这行。 上层再用的话，就不用 target_include_directories 了
# INTERFACE 意味着，使用该lib的都会include CMAKE_CURRENT_SOURCE_DIR，但是 my_lib本身不会 include CMAKE_CURRENT_SOURCE_DIR
target_include_directories(my_lib
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )
```

# 4. Generator Expressions

## interface library
> 通过 interface library 来设置C++版本

```cmake
# TOP
add_library(compiler_flags INTERFACE)
target_compile_features(compiler_flags INTERFACE cxx_std_11)

target_link_libraries(demo1 PUBLIC ${EXTRA_LIBS} tutorial_compiler_flags)
```

```cmake
# subdir
add_library(my_lib my_lib.cpp)
# 加上这个
target_link_libraries(my_lib compiler_flags)
```

# 5. Install and Testing
> 将 demo1 可执行文件 与 my_lib 进行安装
```shell
cmake --install .
cmake --install . --config Release

# cmake变量 CMAKE_INSTALL_PREFIX 用来指定安装root。可通过命令行指定
cmake --install . --prefix "/home/myuser/installdir"
```

```cmake
# subdir

add_library()

set(installable_libs MathFunctions tutorial_compiler_flags)
install(TARGETS ${installable_libs} DESTINATION lib)
install(FILES MathFunctions.h DESTINATION include)
```

```cmake
# topdir

set(CMAKE_INSTALL_PREFIX "/Users/yinkeith/Projects/installed")

install(TARGETS demo1 DESTINATION bin)
install(FILES "${PROJECT_BINARY_DIR}/TutorialConfig.h"
  DESTINATION include
  )
```

# 6. 测试

# 7. 系统自省
> Change implementation based on available system dependencies.
1. 通过在CMakeLists.txt中编译小段代码来确定某特性是否存在
2. 通过target_compile_definitions来设置编译的definitions
3. 代码中通过确定宏是否定义，来决定编译哪个分支

```cmake
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
  #include <cmath>
  int main() {
    std::log(1.0);
    return 0;
  }
" HAVE_LOG)
check_cxx_source_compiles("
  #include <cmath>
  int main() {
    std::exp(1.0);
    return 0;
  }
" HAVE_EXP)


if(HAVE_LOG AND HAVE_EXP)
  #执行这句的话，cmake会帮助定义宏
  target_compile_definitions(MathFunctions
                             PRIVATE "HAVE_LOG" "HAVE_EXP")
endif()
```

```c++
#include <cmath>

double my_func(double x) {

#if defined(HAVE_LOG) && defined(HAVE_EXP)
  double result = std::exp(std::log(x) * 0.5);
  std::cout << "Computing sqrt of " << x << " to be " << result
            << " using log and exp" << std::endl;
#else
  double result = x;

  return result;
}
```

# 8. 添加一个Custom Command 和 Generated File
> 通过一个executable先生成源代码文件，然后再执行后续的编译链接操作

1. 需要一个生成源码文件的可执行文件
2. 生成的源码文件被用来后续的源码编译、链接

```cmake
# 可执行文件target
add_executable(MakeTable MakeTable.cpp)

# 使用构建好的可执行文件执行下述命令
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Table.h
  COMMAND MakeTable ${CMAKE_CURRENT_BINARY_DIR}/Table.h
  DEPENDS MakeTable
  )

# 定义编译链接的lib
add_library(mathfuncs 
        funcs.cpp 
        ${CMAKE_CURRENT_BINARY_DIR}/Table.h)

# 编译时的 include 
target_include_directories(mathfuncs 
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    )
```

# 9. 打包一个安装包
> 打包一个安装包，下载下来，解压就能用。和下载源码下来然后 install 不一样的，省了一步编译操作

```cmake
include(InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set(CPACK_PACKAGE_VERSION_MAJOR "${Tutorial_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${Tutorial_VERSION_MINOR}")
set(CPACK_SOURCE_GENERATOR "TGZ")
include(CPack)
```

```shell
# 在binary directory执行
cpack

# -G 指定generator，生成zip还是 tgz。-C指定配置
cpack -G ZIP -C Debug
```

# 10. static lib 还是 shared lib
通过设置Cmake变量 `BUILD_SHARE_BIBS` 的值来决定 `add_library()` 的默认行为。通常通过 `option` 来指定，这样就可以从命令行空值该值。

```cmake
cmake_minimum_required(VERSION 3.0.0)
project(demo1 VERSION 0.1.0)


configure_file(Config.h.in Config.h)

option(USE_MATH_TOOLS "是否使用math tools" ON)

add_library(compiler_flags INTERFACE)
target_compile_features(compiler_flags INTERFACE cxx_std_17)

# 使用option的话，可以通过命令行来空值该值 cmake .. -DBUILD_SHARED_LIBS=OFF
# option(BUILD_SHARED_LIBS "build using shared libs" ON)
set(BUILD_SHARED_LIBS ON)

add_subdirectory(math_tools)


add_executable(demo1 main.cpp)

target_link_libraries(demo1 PUBLIC math_tools compiler_flags)

target_include_directories(demo1 PUBLIC 
    "${PROJECT_BINARY_DIR}"
    )
```

# 11. Adding Export Configuration
> 添加必要的信息，是的其它CMake项目可以使用我们的项目。从 build 目录、本地安装 或者 安装包

1. install(TARGETS) 不仅需要指定 DESTINATION ，同时也要指定  EXPORT
2. 然后在TOP CMakeLists install export的东西
3. 为了find_package可以找到，还需要生成一个 MathToolsTargetsConfig.cmake
   1. 需要一个模板文件


```cmake
# subdir
add_library(math_tools funcs.cpp)

# 编译的时候将 CMAKE_CURRENT_SOURCE_DIR 添加到 include 搜索路径。
# install的时候将 include 添加到 include 搜索路径
# BUILD_INTERFACE,INSTALL_INTERFACE是个generator expressions 是一种特殊的语法，用于在编译时和安装时根据不同的上下文生成不同的代码
target_include_directories(math_tools 
    INTERFACE 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
     $<INSTALL_INTERFACE:include>
    )

target_link_libraries(math_tools PUBLIC compiler_flags)
set(installable_libs math_tools compiler_flags)
install(TARGETS ${installable_libs}
    EXPORT MathToolsTargets
    DESTINATION lib
)

install(FILES funcs.h DESTINATION include)
```

```cmake
#top dir, 会安装到 PREFIX到该目录下，生成对应的.cmake文件，后续find_library()会用到
install(EXPORT MathToolsTargets
  FILE MathToolsTargets.cmake
  DESTINATION lib/cmake/MathFunctions
)


include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/MathToolsTargetsConfig.cmake"
  INSTALL_DESTINATION "lib/cmake/example"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
  )
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/MathToolsTargetsConfigVersion.cmake"
  VERSION "${Tutorial_VERSION_MAJOR}.${Tutorial_VERSION_MINOR}"
  COMPATIBILITY AnyNewerVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/MathToolsTargetsConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/MathToolsTargetsConfigVersion.cmake
  DESTINATION lib/cmake/MathFunctions
  )

```

```cmake
# Config.cmake.in 模板文件
@PACKAGE_INIT@
include ( "${CMAKE_CURRENT_LIST_DIR}/MathToolsTargets.cmake" )
```

# 12. debug 与 release 包





# 常用cmake变量名总结
* `CMAKE_CURRENT_SOURCE_DIR`: CMakeLists.txt所在源代码目录
* 

# cmake与源码文件协同总结
1. 通过配置文件(一个配置头文件Config.h.in，里面定义了一些占位符，cmake会将其填充，然后生成一个头文件 Config.h)
   1. cmake变量, Config.h.in, @CMAKE_VAR_NAME@, configure_file()
   2. option,  Config.h.in, #cmakedefine, configure_file()
2. 通过传宏名字。这样代码中就可以使用 `#if defined(MACRO1) && defined(MACRO2)` 来进行判断
   1. target_compile_definitions(lib_name PRIVATE "MACRO1" "MACRO2")

# 目标文件依赖项使用范围
> target_** 中的 PUBLIC PRIVATE INTERFACE

# interface target

# 参考资料

[https://cmake.org/cmake/help/latest/guide/tutorial/index.html](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
[https://stackoverflow.com/questions/25676277/cmake-target-include-directories-prints-an-error-when-i-try-to-add-the-source](https://stackoverflow.com/questions/25676277/cmake-target-include-directories-prints-an-error-when-i-try-to-add-the-source)
