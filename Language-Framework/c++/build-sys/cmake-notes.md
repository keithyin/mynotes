# 1 基础

## 最基础版本
**最上层** CmakeLists.txt需要包含
```
# 所需的cmake版本, 非上层的CMakeLists.txt可以不用指定cmake版本
cmake_minimum_required(VERSION 3.0.0)

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

# Usage Requirements for a Library


# 参考资料

[https://cmake.org/cmake/help/latest/guide/tutorial/index.html](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
