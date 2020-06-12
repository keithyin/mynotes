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



# 共享数据保护

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



### mutex

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



### atomic

```c++
// 可能出现脏读的数据， 用 atomic
#include <atomic>

...
int main(){
  std::atomic<int> result(0); //result 是个 atomic 对象，就不会脏读了。
  ...
}
```



### 读写锁

* 使用`boost::shared_mutex`

```c++
#include <boost/thread/shared_mutex.hpp>
mutable boost::shared_mutex rw_mutex;

std::lock_guard<boost::shared_mutex> w_lock(rw_mutex);
std::shared_lock<boost::shared_mutex> r_lock(rw_mutex);
```





## 安全使用mutex

### 正确的锁定和释放 lock_guard

* RAII：可以协助管理锁的锁定和释放
* 构造时候锁定，析构的时候释放锁
* https://en.cppreference.com/w/cpp/thread/lock_tag

```c++
// defer_lock_t	do not acquire ownership of the mutex
// try_to_lock_t	try to acquire ownership of the mutex without blocking
// adopt_lock_t	assume the calling thread already has ownership of the mutex


// 构造时加锁，析构时释放
std::lock_guard<std::mutext> lock(some_mutex);
// std::lock_guard<std::mutex> lock(some_mutex, std::adopt_lock); // 假设 some_mutex已经被加锁，lock_guard只是负责锁的释放而已。

// unique_lock, 可以手动 lock 与 unlock，更加灵活一些， 当然也具有 构造时加锁，析构时解锁
std::unique_lock<std::mutex> uni_lock(some_mutex);//构造加锁
// std::unique_lock<std::mutex> uni_lock(some_mutex, std::defer_lock); // 构造函数中不加锁，之后手动加锁。
uni_lock.unlock(); //手动释放
uni_lock.lock(); // 手动加锁
```



### 初始化过程中保护共享数据

* `std::call_once` : 仅调用一次，线程安全，非常合适用来进行初始化, `once_flag` 是用作是否执行了一次的标记

```c++
std::once_flag conn_once_flag;
std::call_once(conn_once_flag, some_func, some_params);
```

* 局部变量声明为 `static`: static局部变量的初始化为首次执行到时发生。`c++11` 提供了其多线程安全的保证。

```c++
SomeClass& get_instance() {
	static SomeClass instance; //初始化保证是线程安全的, 妥妥的单例模式实现方式，将这个玩意放到类里面就OK了。
  return instance;
}
```







### 如何避免死锁

当一个操作需要两个及以上的锁的时候，就有可能发生死锁

* 假设A操作需要锁a和锁b，B操作也需要锁a和锁b，但是两个操作申请锁的顺序不一致，A是先申请a，B是先申请b。如果有一次，A申请a的同时B也申请了b，那么这两个操作就永远不会执行。死锁了
  * `std::lock` 同时锁定两个或更多的互斥元，该对象仅负责加锁，并不负责解锁，所以还需要与`lock_guard`或者`unique_lock` 一起使用。
  * https://en.cppreference.com/w/cpp/thread/lock

```c++
#include <mutex>
#include <thread>
#include <iostream>
#include <vector>
#include <functional>
#include <chrono>
#include <string>
 
struct Employee {
    Employee(std::string id) : id(id) {}
    std::string id;
    std::vector<std::string> lunch_partners;
    std::mutex m;
    std::string output() const
    {
        std::string ret = "Employee " + id + " has lunch partners: ";
        for( const auto& partner : lunch_partners )
            ret += partner + " ";
        return ret;
    }
};
 
void send_mail(Employee &, Employee &)
{
    // simulate a time-consuming messaging operation
    std::this_thread::sleep_for(std::chrono::seconds(1));
}
 
void assign_lunch_partner(Employee &e1, Employee &e2)
{
    static std::mutex io_mutex;
    {
        std::lock_guard<std::mutex> lk(io_mutex);
        std::cout << e1.id << " and " << e2.id << " are waiting for locks" << std::endl;
    }
 
    // use std::lock to acquire two locks without worrying about 
    // other calls to assign_lunch_partner deadlocking us
    {
        std::lock(e1.m, e2.m);
        std::lock_guard<std::mutex> lk1(e1.m, std::adopt_lock);
        std::lock_guard<std::mutex> lk2(e2.m, std::adopt_lock);
// Equivalent code (if unique_locks are needed, e.g. for condition variables)
//        std::unique_lock<std::mutex> lk1(e1.m, std::defer_lock);
//        std::unique_lock<std::mutex> lk2(e2.m, std::defer_lock);
//        std::lock(lk1, lk2);
// Superior solution available in C++17
//        std::scoped_lock lk(e1.m, e2.m);
        {
            std::lock_guard<std::mutex> lk(io_mutex);
            std::cout << e1.id << " and " << e2.id << " got locks" << std::endl;
        }
        e1.lunch_partners.push_back(e2.id);
        e2.lunch_partners.push_back(e1.id);
    }
    send_mail(e1, e2);
    send_mail(e2, e1);
}
 
int main()
{
    Employee alice("alice"), bob("bob"), christina("christina"), dave("dave");
 
    // assign in parallel threads because mailing users about lunch assignments
    // takes a long time
    std::vector<std::thread> threads;
    threads.emplace_back(assign_lunch_partner, std::ref(alice), std::ref(bob));
    threads.emplace_back(assign_lunch_partner, std::ref(christina), std::ref(bob));
    threads.emplace_back(assign_lunch_partner, std::ref(christina), std::ref(alice));
    threads.emplace_back(assign_lunch_partner, std::ref(dave), std::ref(bob));
 
    for (auto &thread : threads) thread.join();
    std::cout << alice.output() << '\n'  << bob.output() << '\n'
              << christina.output() << '\n' << dave.output() << '\n';
}
```



# 线程同步

* 等待某个事件的发生，一个线程在能完成其任务之前可能需要等待另一个线程完成任务



##future

* 使用`future`等待一次性事件

* 两个`future`: `<future>`头文件，`std::future<>, std::shared_future<>`, 这两个所对应的语义是`std::unique_ptr, std::shared_ptr`. 独享事件或者共享事件
* `std::async` 的行为是可以手动控制的。
  * `std::launch::async`: 表示该函数必须在自己的线程上运行（默认）
  * `std::launch::deferred`: 函数的调用会延迟到 `future` 调用 `wait` 或者` get`
  * `std::async(std::launch::deferred, func, params)`

```c++
#include <future>
int find_answer_to_ltuae();
void do_other_stuff();

int main() {
  // 开启一个异步线程，函数的返回值会给到 the_answer
  std::future<int> the_answer = std::async(find_answer_to_ltuae);
  do_other_stuff();
  // 当调用 get 的时候， 会阻塞等待 异步线程完成。
  int answer = the_answer.get();// get之后再get就没有值了，shared_future也是一样
}
```

* `class std::packaged_task<>` : 将异步函数执行和返回的`future`封装起来，可以拿着随便移动（`std::move`）。 
  * 对象被调用的时候是异步执行

```c++
std::packaged_task<int()> task{task_func};
std::future<int> task_future = task.get_future();
```

* `promise` & `future`, 这是一对，当 `promise` 设置值的时候，`future` 就会收到值，就像是 `golang` 中 `channel`

```c++
std::promise<int> ch_in;
std::future<int> ch_out = ch_in.get_future();
ch_in.set_value(10);
std::cout << ch_out.get() << std::endl;
```

* 为 `future` 保存异常
  * 当 `async` 调用的函数产生异常，该异常会被保存在 `future`中，在调用 `get` 的时候触发。`packaged_task`同理
  * 使用 `promise`的时候，可以`promise.set_exception()`，这样，该异常也会保存在 `future` 中，调用`get`的时候触发

```c++

```







## condition_variable

[https://www.cnblogs.com/haippy/p/3252041.html](https://www.cnblogs.com/haippy/p/3252041.html)

> A *condition variable* is an object able to block the calling thread until *notified* to resume.
>
> 当 cv的 wait 方法被调用时，它使用 `unique_lock (over mutex)` 来锁住线程。直到其它线程 调用 `notification method` 来将其唤醒。

```c++
// condition_variable example
#include <iostream>           // std::cout
#include <thread>             // std::thread
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable

std::mutex mtx;
std::condition_variable cv;
bool ready = false;

void print_id (int id) {
  // unique_lock 这个创建对象的时候，就已经调用了 mtx.lock() 
  std::unique_lock<std::mutex> lck(mtx); 
  while (!ready)  // 如果标志位不为 true ，则等待！！！ 由 cv.wait(lck) 阻塞
    cv.wait(lck); // 当 mtx locked 时， 该函数会 调用 lck.unlock() 释放锁。
    // 在被唤醒时， lck 被设置为 进入 wait 之前的 状态！！！
  std::cout << "thread " << id << '\n';
}

void go() {
  std::unique_lock<std::mutex> lck(mtx);
  ready = true;
  cv.notify_all();
}

int main ()
{
  std::thread threads[10];
  // spawn 10 threads:
  for (int i=0; i<10; ++i)
    threads[i] = std::thread(print_id,i);

  std::cout << "10 threads ready to race...\n";
  go();                       // go!

  for (auto& th : threads) th.join();

  return 0;
}
```



## unique_lock

>  unique_lock内部持有mutex的状态：locked,unlocked。unique_lock比lock_guard占用空间和速度慢一些，因为其要维护mutex的状态。
>
>  构造函数中加锁，
>
>  析构函数中解锁。 当然也可以灵活操作。

```c++
// unique_lock example
#include <iostream>       // std::cout
#include <thread>         // std::thread
#include <mutex>          // std::mutex, std::unique_lock

std::mutex mtx;           // mutex for critical section

void print_block (int n, char c) {
  // critical section (exclusive access to std::cout signaled by lifetime of lck):
  std::unique_lock<std::mutex> lck (mtx);
  for (int i=0; i<n; ++i) { std::cout << c; }
  std::cout << '\n';
}

int main ()
{
  std::thread th1 (print_block,50,'*');
  std::thread th2 (print_block,50,'$');

  th1.join();
  th2.join();

  return 0;
}
```

```
Possible output (order of lines may vary, but characters are never mixed):
**************************************************
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
```



# std::condition_variable

**头文件**

> `#include <condition_variable> `

**condition_variable**

[https://www.cnblogs.com/haippy/p/3252041.html](https://www.cnblogs.com/haippy/p/3252041.html)

> A *condition variable* is an object able to block the calling thread until *notified* to resume.
>
> 当 cv的 wait 方法被调用时，它使用 `unique_lock (over mutex)` 来锁住线程。直到其它线程 调用 `notification method` 来将其唤醒。

```c++
// condition_variable example
#include <iostream>           // std::cout
#include <thread>             // std::thread
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable

std::mutex mtx;
std::condition_variable cv;
bool ready = false;

void print_id (int id) {
  // unique_lock 这个创建对象的时候，就已经调用了 mtx.lock() 
  std::unique_lock<std::mutex> lck(mtx); 
  /* 这个位置一般放置访问共享数据的代码*/
  while (!ready)  // 如果标志位不为 true ，则等待！！！ 由 cv.wait(lck) 阻塞
    cv.wait(lck); // 当 mtx locked 时， 该函数会 调用 lck.unlock() 释放锁。
    // 在被唤醒时， lck 被设置为 进入 wait 之前的 状态！！！
  std::cout << "thread " << id << '\n';
}

void go() {
  std::unique_lock<std::mutex> lck(mtx);
  /*这个位置一般放置 访问共享数据的代码*/
  ready = true;
  cv.notify_all();
}

int main ()
{
  std::thread threads[10];
  // spawn 10 threads:
  for (int i=0; i<10; ++i)
    threads[i] = std::thread(print_id,i);

  std::cout << "10 threads ready to race...\n";
  go();                       // go!

  for (auto& th : threads) th.join();

  return 0;
}
```



**wait(lck)** : 这个 lck 的意义就是，如果 wait 开始向下执行，就将 lck 设置成 lock 状态

* 调用时，如果 `lck` 为 `locked` 则， 执行`lck.unlock()`
* 被 notify 时： 会调用 `lck.lock()` 使得 `lck` 回到 调用 `wait` 之前的状态

**wait(lck, pred)**

* 只有当 pred 条件为 false 时调用 wait() 才会阻塞当前线程，并且在收到其他线程的通知后只有当 pred 为 true 时才会被解除阻塞

**notify_one**

* 唤醒某个等待(wait)线程。如果当前没有等待线程，则该函数什么也不做，如果同时存在多个等待线程，则唤醒某个线程是不确定的



**notify_all**

* 唤醒所有的等待(wait)线程。如果当前没有等待线程，则该函数什么也不做。



**对于 notify 之后，wait 会调用 lck.lock 的 验证代码**

```c++
// condition_variable example
#include <iostream>           // std::cout
#include <thread>             // std::thread
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable

std::mutex mtx;
std::condition_variable cv;
bool ready = false;

void print_id (int id) {
    // unique_lock 这个创建对象的时候，就已经 mtx.lock 了
    // 其它线程就阻塞在了 unique_lock 的构造函数中！！！
    std::unique_lock<std::mutex> lck(mtx);
    while (!ready) {  // 如果标志位不为 true ，则等待！！！ 由 cv.wait(lck) 阻塞
        std::cout << "i am thread " << id << " waiting here" << std::endl;
        cv.wait(lck); // 使用 unique_lock (over mutex) 来将其锁住
    }
    for (int i=0; i < 10; ++i)
        std::cout << "thread " << id<< " value:" << i <<std::endl;
}

void go() {
    std::unique_lock<std::mutex> lck(mtx);
    ready = true;
    cv.notify_all();
}

int main ()
{
    std::thread threads[10];
    // spawn 10 threads:
    for (int i=0; i<10; ++i)
        threads[i] = std::thread(print_id,i);

    std::cout << "10 threads ready to race...\n";
    go();                       // go!

    for (auto& th : threads) th.join();

    return 0;
}

```







## 参考资料

[https://www.cnblogs.com/haippy/p/3237213.html](https://www.cnblogs.com/haippy/p/3237213.html)







## 参考资料

[http://www.bogotobogo.com/cplusplus/multithreaded4_cplusplus11.php](http://www.bogotobogo.com/cplusplus/multithreaded4_cplusplus11.php)

[http://www.bogotobogo.com/cplusplus/C11/7_C11_Thread_Sharing_Memory.php](http://www.bogotobogo.com/cplusplus/C11/7_C11_Thread_Sharing_Memory.php)

[http://www.cplusplus.com/reference/condition_variable/condition_variable/](http://www.cplusplus.com/reference/condition_variable/condition_variable/)

[http://www.cplusplus.com/reference/mutex/unique_lock/](http://www.cplusplus.com/reference/mutex/unique_lock/)

[http://blog.csdn.net/liuxuejiang158blog/article/details/17263353](http://blog.csdn.net/liuxuejiang158blog/article/details/17263353)





