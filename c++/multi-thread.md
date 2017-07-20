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

  // 阻塞当前线程，直至 t1 线程运行完毕，
  // 如果主线程已经运行结束，但是其它线程还在跑的话，会导致 运行时 错误。
  t1.join();

  return 0;
}
```

然后在 linux 上，可以使用以下命令来编译 此文件

```shell
g++ -o file_name -std=c++11 -pthread file_name.cc 
```



**如果线程调用的函数有参数，那该怎么办呢？**

```c++
#include <iostream>
#include <thread>

static const int num_threads = 10;

void call_from_thread(int tid) {
  std::cout << "Launched by thread " << tid << std::endl;
}

int main() {
  std::thread t[num_threads];

  for (int i = 0; i < num_threads; ++i) {
    t[i] = std::thread(call_from_thread, i);//执行线程的时候传入就可以了。
  }

  std::cout << "Launched from the main\n";
 
  for (int i = 0; i < num_threads; ++i) {
    t[i].join();
  }

  return 0;
}
```



## 同步问题

> 涉及到的两个头文件 mutex , atomic(原子)

看下面代码：

```c++
#include <iostream>
#include <thread>
#include <vector>
  
void dot_product(const std::vector<int> &v1, const std::vector<int> &v2, int &result, int L, int R){
     for(int i = L; i < R; ++i){
         result += v1[i] * v2[i];
     }
 }
 
 int main(){
     int nr_elements = 100000;
     int nr_threads = 4;
     int result = 0;
     std::vector<std::thread> threads;
 
     //Fill two vectors with some constant values for a quick verification
     // v1={1,1,1,1,...,1}
     // v2={2,2,2,2,...,2}
     // The result of the dot_product should be 200000 for this particular case
     std::vector<int> v1(nr_elements,1), v2(nr_elements,2);
 
 
     //Launch nr_threads threads:
     for (int i = 0; i < nr_threads; ++i) {
       	 // 这里要注意的是，函数参数为 引用的时候，一定要加上 std::ref(..) 否则会报错。
         threads.push_back(std::thread(dot_product, std::ref(v1), std::ref(v2), std::ref(result), i*25000, (i+1)*25000));
     }
 
 
     //Join the threads with the main thread
     for(auto &t : threads){
         t.join();
     }
 
     //Print the result
     std::cout<<result<<std::endl;
 
     return 0;
}
// 运行结果为
// 第一次 121110 
// 第二次 200000
```

为什么两次结果不同呢，这就考虑到 多线程同步的问题了。

```c++
result += v1[i] * v2[i];
// 可能会造成 脏读问题。
```



**解决方法一：mutex**

```c++
#include <iostream>
#include <thread>
#include <vector>
#include <mutex> 

static std::mutex barrier;  // 全局 mutex 对象
void dot_product(const std::vector<int> &v1, const std::vector<int> &v2, int &result, int L, int R){
     int partial_sum = 0;
     for(int i = L; i < R; ++i){
         partial_sum += v1[i] * v2[i];
     }
  	 // 用 mutex 的对象来进行同步， 
  	 // 使用 lock_guard 或 或者 barrier.lock() ; barrier.unlock();
  	 // lock_gurad 在构造函数中 lock， 在析构函数中 unlock()
     std::lock_guard<std::mutex> block_threads_until_finish_this_job(barrier);
     result += partial_sum;
}
```



**解决方法二：atomic**

```c++
// 可能出现脏读的数据， 用 atomic
#include <atomic>

...
int main(){
  std::atomic<int> result(0); //result 是个 atomic 对象，就不会脏读了。
  ...
}
```



## join detach

一旦线程启动，我们可以通过 `join` 让代码知道我们是想 等待这个线程执行完，或者通过 `detach` 告诉代码让这个线程自己玩。如果没有显式的做 `join` 或 `detach` 工作的话，`std::thread` 对象会随着 主线程的执行完毕而被销毁（这时 `std::thread` 代表的线程可能还没有执行完，会报错）。



**join：** 阻塞线程运行

```c++
#include <iostream>
#include <thread>

void foo() { std::cout << "foo()\n"; }
void bar() { std::cout << "bar()\n"; }

int main()
{
	std::thread t([]{
		        foo();
			bar();						 
	                });
    t.join();  // 主线程会阻塞在这个位置，直到t线程执行完
	return 0;
}
```

**detach：** daemon thread（不等待线程执行完）

当线程 `detach` 后，线程的执行由 `c++ Runtime Library` 控制。



## 线程数量

* `std::thread::hardware_ concurrency()` 会返回 `cpu` 的核心数





## 参考资料

[http://www.bogotobogo.com/cplusplus/multithreaded4_cplusplus11.php](http://www.bogotobogo.com/cplusplus/multithreaded4_cplusplus11.php)

[http://www.bogotobogo.com/cplusplus/C11/7_C11_Thread_Sharing_Memory.php](http://www.bogotobogo.com/cplusplus/C11/7_C11_Thread_Sharing_Memory.php)



