## cmake 全局变量

* `PROJECT_SOURCE_DIR`: contains the full path to the root of your project source directory, i.e. to the nearest directory where `CMakeLists.txt` contains the `PROJECT()` command
* `CMAKE_SOURCE_DIR`: this is the directory, from which cmake was started, i.e. the top level **source directory**



* `CMAKE_BINARY_DIR` : if you are building in-source, this is the same as `CMAKE_SOURCE_DIR`, otherwise this is the top level directory of your **build tree**

* `EXECUTABLE_OUTPUT_PATH`: set this variable to specify a common place where CMake should put all executable files (instead of `CMAKE_CURRENT_BINARY_DIR`)

* `LIBRARY_OUTPUT_PATH`: set this variable to specify a common place where CMake should put all libraries (instead of `CMAKE_CURRENT_BINARY_DIR`)

* `PROJECT_NAME` : the name of the project set by `PROJECT()` command.

  * 仅仅是项目名称而已，和可执行文件名没有直接关系




* `add_executable(hello ${PROJECT_SOURCE_DIR}/main.cc)`
  * 编译一个可执行文件
* ​




