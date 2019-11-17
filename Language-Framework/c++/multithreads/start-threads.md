# 开始一个线程

```c++
#include<thread>
#include<iostream>

void hello(){
  std::cout<<"hello from thread" << std::endl;
}

int main(){
  std::thread t1(hello);
  t1.join(); // main thread 需要等待 t1 thread 执行完毕。
  return 0;
}
```

* 开始线程：在 C++ 中，开始一个线程非常简单，当创建一个线程实例的时候，线程就自动的开始了。
* 显式的等待其结束: 
  * `.join()`, 强制当前线程等待 调用线程执行完毕。如果不使用这个 方法的话，结果是未定义的，因为一旦主线程结束，该进程下的其它线程也会强制结束。
* 让其自生 **一起灭**
  * `.detach()`: detach 了之后，就变成了 daemon thread，**主线程结束，detach()的线程也会结束**



**Caveat**

* Once our thread started, we should let the code know if we want to wait for it to finish by **joining** with it or leave it to run on its own by **detaching** it. Our program may be terminated before the **std::thread** object is destroyed if we don't do anything explicitly.
* `.detach()` 的时候一定要保证，**线程中访问的数据是有效的**！！！
* 在 `.join(), .detach()` 的时候，一定要保证 线程是 `.joinable()` 的。



**如何传递参数**

* 传值
* 传引用 (`std::ref(i)`) 
* 类方法内部调用类方法 `作为多线程的初始函数` , 要传 `this` 为啥?

```c++
void CallableObj(double d, std::string const& s);
// 直接传到 thread 的 构造函数中就可以了。
std::thread t(CallableObj, 3.14, "pi");
```

* Note that even though **CallableObj** takes a **string** as the second parameter, the string literal is passed as a **char const\*** and converted to a **string** only in the context of the new thread

```c++
std::thread t(CallableObj, 3.14, "pi");
std::thread &&t_ = std::move(t);
t.joinable(); // False
```

**如果可调用对象的参数是引用，应该怎么做呢？**

```c++
void func(int &i){
  std::cout<<i<<std::endl;
}
int i = 10;
std::thread t(func, std::ref(i));
```

* 当传递 reference 给函数时，需要注意的事情是 **reference 变量的生存周期**



# 如何区分线程

* C++ 中每个线程都有一个唯一 id，可以通过 `std::this_thread::get_id()` 获得。
* 可以通过 `std::this_thread::` 里面的函数操作当前 `thread` 的行为。

```c++
#include <thread>
#include <iostream>
#include <vector>

void hello(){
    std::cout << "Hello from thread " << std::this_thread::get_id() << std::endl;
}

int main(){
    std::vector<std::thread> threads;

    for(int i = 0; i < 5; ++i){
        threads.push_back(std::thread(hello));
    }

    for(auto& thread : threads){
        thread.join();
    }

    return 0;
}
```



# 保护共享数据, shared memory co-currency

> 共享数据的保护由 mutex(互斥量) 实现
>
> * `#include<mutex>` : `class std::mutex, class std::lock_guard`

* `std::mutex` 互斥量
* `std::lock_guard` : 互斥量的资源管理类, 创建时会调用 `mutex::lock()` , 析构时自动调用 `mutex::unlock()`, 减少了手动 `mutex` 资源管理的麻烦. 

需要注意的几个点:

* 被互斥量保护的数据结构, 千万不要一不小心将其 引用传出来.
* 仅仅因为是在 数据结构的 某些操作上是安全的, 依旧可能导致 竟态发生
  * 比如: 列表的 push, pop, top 单独操作都很安全. 但是将 top 和 pop 合并成一个 操作的时候可能就会发生危险, 因为不确定在 top 和 pop 之间会不会出现一个 push 操作, 导致 pop 出来的值并不是 top 时候的值.
* `std::unique_lock(mutex, std::defer_lock)` 
* `std::lock_guard(mutex, std::adopt_lock)` 

> unique_lock VS lock_guard, 都是管理 mutex 资源的, 但是 unique_lock提供了更大的灵活性

* `std::call_once`  (资源的延迟初始化)
  * 多线程情况下的资源初始化 (保证资源只初始化一次(函数只调用一次))

```c++
std::shared_ptr<some_resource> resource_ptr;
std::once_flag resource_flag; // call_once 的标记

void init_resource(){
	resource_ptr.reset(new some_resource);
}

void func() {
  // 多线程情况下resource_flag 和 call_once保证 init_resource 只会被调用一次
	std::call_once(resource_flag, init_resource);
  resource_flag->do_something();
}
```

* static 局部变量的初始化:
  * 时机: 第一次调用的时候
  * C++11: 保证在多线程的情况下, 保证正确的初始化 (在一个线程上初始化, 并且初始化未完成之前, 其它执行 该 static 语句的代码会被阻塞).

```c++
class Myclass;
Myclass& getMyClass(){
	static Myclass myclass; //保证初始化是安全的.
  return myclass;
}
```

* 保护很少更新的数据结构 (read-write mutex)
  * 只有读的时候 是不会加锁的



# 线程同步

> 相互通知一下.

线程同步的方法

* 可以使用 flag + 定时轮询的方式
* 使用 `#include <condition_variable>` 中的 来进行主动通知
  * 使得正在等待工作的线程休眠, 直到有数据要处理

* `condition_variable::wait`
  * 非通知情况下
    * 条件判断 (首先在执行 wait 之前是加了锁的, 为了保证条件判断的正确性)
    * 条件满足: 1)继续往后走
    * 条件不满足: 1)释放 锁(为了生产者可以正常生产数据), 2) 阻塞 (这时候 锁是释放的, 阻塞在 condition_variable 上)
  * 被通知
    * 1) 被唤醒, 2) 给互斥元加锁, 3) 检查条件, 
    * 如果条件满足, 1)返回
    * 不满足: 1) 释放锁, 2)阻塞

```c++
std::mutex mux;
std::queue<data_chunk> data_queue;
std::condition_variable data_cond;

void data_preparation_thread() {
  while (more_data_to_prepare()) {
    data_chunk const data = prepare_data();
    std::lock_guard<std::mutex> lk(mut);
    data_queue.push(data);
    data_cond.notify_one();
  }
}

void data_processing_thread() {
	while (true) {
    // 这边使用 unique_ptr 是为了提供更多的灵活性
    // lock, try_lock, unlock, 这些 unique_lock 都有
  	std::unique_lock<std::mutex> lk(mut);
    data_cond.wait(lk, [](){return !data_queue.empty()});
    data_chunk data = data_queue.front();
    data_queue.pop();
    lk.unlock();
    process(data);
  }
}
```

* 等待一次性事件 (不需要循环等待的事件) `class std::future, std::async()`

```c++
std::future<int> the_val = std::async(func, arg);// 异步执行, 不会阻塞
// 这里可以做一些其它的事情
the_val.get(); //阻塞, 直到异步任务完成.
```





# Return values from threads

* Promise / Future  ： `<future>`
* Promise : 输入端
* Future：输出端



`chrono`: 用来处理时间相关

```c++
void promise_set(promise<string> &&pms){
  pms.set_value(string("hello world"));
}

void test_promise_future(){
  promise<string> pms; // 会开辟一段共享空间
  future<string> ftr = pms.get_future();
  thread t(promise_set, std::move(std));
  cout<< "blocking ..." << endl;
  
  // 如果共享空间没有被 set_value，这个地方就会被 block 住
  // 如果共享空间被 set 了， 这儿就可以往下运行了
  // get() 也是获取 共享空间的 值
  string res = ftr.get();
  cout <<res <<endl;
  t.join();
}
```



**另一种操作：async**

```c++
// 更少的代码，达到和 promise / future 相同的效果
string func(){
  std::string str("hello world");
  return str;
}

int main(){
  // ftr 的析构函数 保证 线程会在最后 join 一下
  // 当然，func 也可以不返回值，这样 ftr 就 get 出来空
  future ftr = std::async(func);
  string str = ftr.get();
  cout << str << endl;
  reurn 0;
}
```







