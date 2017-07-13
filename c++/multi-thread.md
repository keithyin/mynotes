# c++ 多线程编程

> c++ 11 才引入了多线程编程，在之前只能用系统相关的库进行多线程编程



## thread 头文件

> 多线程部分在 thread 头文件中

```c++
// file_name.cc
#include <iostream>
#include <thread>

// 将在线程中调用这个函数

void call_from_thread() {
  std::cout << "Hello, World" << std::endl;
}

int main() {
  // 启动一个线程
  std::thread t1(call_from_thread);

  //Join the thread with the main thread
  t1.join();

  return 0;
}
```

然后在 linux 上，可以使用以下命令来编译 此文件

```shell
g++ -o file_name -std=c++11 -pthread file_name.cc 
```

