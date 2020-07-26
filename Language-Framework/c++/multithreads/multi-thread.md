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

然后在 `linux` 上，可以使用以下命令来编译 此文件

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



### mutex & recursive_mutex

* `mutex`: 锁（互斥量）
* `recursive_mutex`: 递归锁，同一个线程内，可以对其进行多次加锁。释放的时候，也需要进行多次解锁
  * 常用于递归代码中。所以名字为`recursive_mutex`

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

### 读写锁

* 使用`boost::shared_mutex`
  * 写互斥，读共享
  * 写的时候使用`std::lock_guard`, 读的时候使用`boost::shared_lock`

```c++
#include <boost/thread/shared_mutex.hpp>
mutable boost::shared_mutex rw_mutex;

std::lock_guard<boost::shared_mutex> w_lock(rw_mutex);
boost::shared_lock<boost::shared_mutex> r_lock(rw_mutex);
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



## 安全使用mutex

* `RAII` 方式管理互斥量
  * `lock_guard`
  * `unique_guard`

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

### unique_lock

>  unique_lock内部持有mutex的状态：locked,unlocked。unique_lock比lock_guard占用空间和速度慢一些，因为其要维护mutex的状态。
>
>  构造函数中加锁，析构函数中解锁。 
>
>  当然也可以灵活操作。可以在创建对象的时候加个其他参数，这样在构造函数中就不会加锁了。

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
  * 为了避免死锁，常见的解决方案是 按照相同的顺序 锁定两个互斥元。这个方案有时候很直观，有时候却不是。所以我们需要一个更好的解决方案。
  * 将多个加锁的操作合并为原子操作： `std::lock` 同时锁定两个或更多的互斥元，该对象仅负责加锁，并不负责解锁，所以还需要与`lock_guard`或者`unique_lock` 一起使用。
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
        // adopt_lock 告诉 lock_guard 不要在构造函数中加锁。
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

## future

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

## condition_variable

* `condition_variable`: 条件变量，常用来建模生产者消费者模型。
* `condition_variable` 语义上提供两个功能：
  1. `wait` 的阻塞， 和 `notify` 的通知
  2. 在`wait` 的时候顺便帮着管理了一下 `mutex`

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
/*
	作为一个消费者应该做的事情就是，
	1. 加锁，准备访问共享数据
	2. 判断共享数据是否可以访问，
		如果可以访问就访问，
		不可以访问就wait（阻塞着等通知），，这里就用到共享变量了（condition_variable）
		如果有通知了之后（需要继续访问共享数据）
*/
void consumer (int id) {
  // unique_lock 这个创建对象的时候，就已经调用了 mtx.lock() 
  std::unique_lock<std::mutex> lck(mtx); 
  /* 这个位置一般放置访问共享数据的代码，
  	因为这里一般放置的是
  */
  while (!ready)  // 如果标志位不为 true ，则等待！！！ 由 cv.wait(lck) 阻塞
    cv.wait(lck); // 当 mtx locked 时， 该函数会 调用 lck.unlock() 释放锁。
    // 在被唤醒时， lck 被设置为 进入 wait 之前的 状态！！！
  std::cout << "thread " << id << '\n';
}

/*
  作为一个生产者
  1. 加锁， 准备生产
  2. 生产完成，（通知等待在该 condition_variable 上的线程，然后解锁）
  		私以为：这个 通知 和解锁操作 顺序是啥不影响。
*/
void productor() {
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

* 调用时：根据上述的`consumer`功能，此时 `lck` 必定是 `locked`， 执行`lck.unlock()`。
  * 为什么被设计成执行`unlock`: 原因：既然数据没有准备好，那就把锁让出来 给 生产者来用
* 被 notify 时： 会调用 `lck.lock()` 
  * 为什么会设计成执行`lock`: 原因：被notify之后，当然是要操作共享数据啦，所以这里设计成为这样也是合理

**wait(lck, pred)**

* 调用时：
  1. 检查 pred 条件，如果为true，直接返回，啥也不操作（不解锁lock，不阻塞）
     * 如果条件为false，先解锁互斥元，然后阻塞该线程（等着被notify）
* notify时
  1. 解除阻塞
  2. 互斥元加锁
  3. 检查条件，如果条件为 `true` 返回
     * 如果条件为 `false` ： 解锁互斥元，然后阻塞该线程（等着被notify）

**notify_one**

* 唤醒 **当前** 某个等待(wait)线程。如果当前没有等待线程，则该函数什么也不做，如果同时存在多个等待线程，则唤醒某个线程是不确定的

**notify_all**

* 唤醒 **当前** 所有的等待(wait)线程。如果当前没有等待线程，则该函数什么也不做（因为什么也做不了）。



# 原子操作

* 原子类型可以在 `<atomic>` 头文件中找到，**在这种类型上的所有操作都是原子的！！**

  * ps: 使用 `mutex` 可以将某些操作 **模拟成原子** 的，但是原子类型上的操作 本身就是原子的。（原子类型具体怎么实现的先不用考虑）

* 几乎每个原子类型都有一个 `is_lock_free()` 方法，这个方法返回 true 说明该原子类型是无锁的，如果返回false，说明该原子类型是通过 锁 模拟出来的？

  * ps：这里讲的原子类型: std::atomic<> 类模板特化出来的任何类型 和 `std::atomic_flag`

* `std::atomic_flag`: 只有以下两个方法

  * `test_and_set()`
  * `clear()`
  * ps: 注意，该类没有 `is_lock_free` 哦， 因为它一定是 `lock_free` 的。

* 标准的原子类型是 不可复制且不可赋值的（因为没有 复制构造函数 和 赋值运算符）

  * 但是，他们支持从**相应的 内置类型 进行隐式转换并赋值**。
  * 适当的地方还提供： `=, +=, -=, *=, |=, ++, -- ...etc` 。这些运算符还有对应的方法 `fetch_add, fetch_or, ...etc`

* 原子操作的 赋值操作符 和 与其相对应的 方法 返回的值 要么是 存储的值 或者 是之前的值。因为返回的不是引用（c++许多赋值运算符的返回值是引用。），所以就避免了可能存在的 数据竞争的问题。

* `std::atomic<>` 并不仅仅只是一堆特化。该模板还支持用户自定义类型。由于他是一个泛化的模板，所以其操作只限 `load(), store(), exchange(), compare_exchange_weak(), compare_exchange_strong()` 

  * ps: `std::atomic<int>` 因为是 特化的，所以会有比 `std::atomic<>` 更多的方法！！！

* `std::atomic<>`  类型的每个操作都可以指定一个额外的参数：`memory-order argument` ，这个参数可以指定一个我们所期望的 内存顺序语义。

  |                                |                        |                        |                        |                        |                        |                        |
  | ------------------------------ | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
  | `Store operations`             | `memory_order_relaxed` | `memory_order_release` | `memory_order_seq_cst` |                        |                        |                        |
  | `Load operations`              | `memory_order_relaxed` | `memory_order_consume` | `memory_order_acquire` | `memory_order_seq_cst` |                        |                        |
  | `Read-Modify-Write operations` | `memory_order_relaxed` | `memory_order_consume` | `memory_order_acquire` | `memory_order_release` | `memory_order_acq_rel` | `memory_order_seq_cst` |

  * 默认情况下：对于所有的操作，都是 `memory_order_seq_cst` 。
  * 需要注意的是，这个 `memory-order` 和之前提到的 原子上的操作都是原子的 是两个东西。两个是用来做不同事情的。

* 为什么原子类型没有 赋值 和 拷贝操作呢？原因如下

  * 原子类型上的操作都是原子的
  *  赋值 和 构造 操作包含两个 原子对象，需要从一个原子中读，然后赋值给另一个原子。因为这是在**两个原子对象** 上的 **两个不同操作**，这种组合不能被原子化。 （为啥不能呗原子化呢？）
  * 所以原子类型没有拷贝 和 赋值 操作



## 使用 `std::atomic_flag` 实现一个自旋锁

```c++
class spinlock_mutex {
    std::atomic_flag flag_;
public:
    spinlock_mutex(): flag_(ATOMIC_FLAG_INIT){}
    void lock() {
        // test_and_set, 将 flag 的值设置 为 true，并返回之前的值
        while (flag.test_and_set(std::memory_order_acquire)){}
    }
    void unlock() {
        //  将 flag 的值设置 为 false
        flag.clear(std::memory_order_release);
    }
}
```

* 以上实现的锁可以在 `std::lock_guard` 中使用，当我们看到 `memory-order` 语义的时候，就能够明白这部分代码为什么能满足 `mutex` 的语义！！！
* `mutex` 的语义是啥呢？mutex 内修改的值，对于后进入该mutex 的线程可见！！

## `std::atomic<bool>`

```c++
std::atomic<bool> b(true);
b = false;
bool x = b.load(std::memory_order_require);
b.store(true);
x = b.exchange(false, std::memory_order_acq_rel);
```

* 新操作：`storing a new value (or not) depending on the current value`
  * `compare_exchange_weak(), compare_exchange_strong()`



## Memory Order

### 程序的乱序执行

* 程序是乱序执行的

  * 同一线程中，彼此 **没有依赖关系** 的指令有可能会被乱序执行。有依赖关系的还是顺序执行的。
  * 但是在单线程程序中，乱序执行并不会对程序执行的结果造成什么影响
  * 这里主要关注读写执行，因为他们会产生外部可见的影响

* 乱序执行的原因：

  * 编译器优化：
    * 编译器优化的目标保证的是 单线程运行的 正确性。
    * 编译器优化的时候可能会打乱 源码 的指令顺序
  * 处理器乱序执行：防止 l1 cache miss 导致 cpu 等待太久
    * l1 cache 读取数据一般是 一个 cycle
    * memory 读取数据一般就得是 100 个 cycle 了
  * 存储系统
    * 一般认为：一旦数据到 l2 cache，那么所有的cpu看到的数据就是一致的。
    * cpu写指令结束后，数据只是放到了 store buffer 中，还没有进入 l2 cache，意味着不同cpu看到的数据可能不一致。
  * on-chip network

  ### 乱序执行的后果

  ```c++
  // 初始 x = y = 0
  // 线程 1
  
  {
      x = 1;  // 这里 x = 1，y=1 的顺序可能会被编译器（或者CPU）调换
  	y = 1;
  }
  
  // 线程2
  
  if (y == 1) {
      assert (x == 1); //这里assert 可能会失败
  }
  ```

  * if 的条件也是可以和之前的 代码 `out-of-order` 的

  ```c++
  // 源码
  x = 1;
  if (y == 0) {
      do_something
  }
  
  // 编译器优化后的代码（可能是）
  register = (y == 0);
  x = 1;
  if (register) {
      
  }
  ```

  * if 里面的代码也有可能搞到 if 的外面去。。。

  ```c++
  // 源码
  if (y == 0) {
      read x;
  }
  
  // 编译器优化后（可能是）
  read x;
  if (y == 0) {
      x;
  }
  ```

  

  * 乱序执行不可避免
  * 如果读写指令所涉及的变量不是 线程之间共享变量，那么乱序执行不会产生坏的影响
  * 如果读写指令所涉及的变量是 线程间共享变量，程序员则需要告诉编译器和处理器。告诉的方式就是 锁 或者 原子操作
    * 当临界区包含多条指令时，使用 锁
    * 当临界区只包含一个整数 或者 指针操作时，使用原子变量



### Acquire & Release 一致性

* `acquire` ：表示的是 获取锁操作
  * 之后的 **所有指令（读写）**，不会早于该指令执行 （这是为了保证 **源码中的临界区** 为 **执行时的临界区**）
  * 后面的指令一定会乖乖的呆在后面
* `release`: 表示的是 释放锁 操作
  * 之前的**所有指令都已经执行完（尤其是写指令）** ，并且已经 **全局可见** 
  * 前面的指令 一定会在 `release` 之前执行完。（这里是为了保证，临界区的修改结束之后，全局可见。）
* `acquire & release`  编译器级别的实现
  * acquire 和 release 操作的内部实现需要利用 memory barrier
  * acquire 和 release 操作由汇编语言编写，因此可以排除编译器优化的影响，同时通过汇编语言也可以方便的嵌入 memory barrier 指令
  * 当编译器看到 memory barrier 时，不会把 acquire 后面的指令挪到 acquire 前面，也不会把 release前面的指令移动到 release 后面。
  * 编译器不能把一个函数调用后面的指令挪到该函数调用的前面，也不能将一个函数调用前面的指令挪到该函数调用的后面，因为编译器不知道该函数调用内部是否使用了 memory-barrier指令。
* `acquire & release`  cpu级别的实现
  * PowerPC 的 lwsync 指令是 memory barrier 指令，其工作原理是堵在 处理器流水线的入口，不让后续指令进入流水线，直到前面已经进入流水线的指令完成，并且 store buffer 清空。
  * 逻辑上看，lwsync 保证它前面的指令不会被挪到它后面，它后面的指令不会被挪到它前面。因此是一个双向的 memory-barrier
  * cpu流水线第一级是 取指令，当取到 lwsync 指令的时候，就堵着不让后面的指令进来。

```c++
/*cpu memory-barrier 如何 实现 acquire-release*/
// 线程 1
acquire;
write x;
lwsync;
Ready = 1;

// 线程2
While(Ready != 1) {}
lwsync;
read x;
release;
```

* 单独的 `memory barrier` 指令代价太大，原因如下
  * 无论是编译器导致的 out-of-order 还是 cpu导致的 out-of-order 都是为了更好的优化代码
  * `memory-barrier` 导致的后面的不能前，前面的不能往后，限制了编译器和CPU优化的能力

* 合并的 `acquire` 和 `release` (解决 单独的 memory-barrier 指令代价大的问题)
  * Inter IA64处理器将 Memory-Barrier指令和 `Ready` 读写指令进行合并，提供了 带 acquire 语义的读指令 `ld.acq` （`acquire_load`） 和 带 release 语义的写指令 `st.rel` （`release store`） 。
  * 这样的好处是 增加了 编译优化的可能性。
  * 锁的底层实现实际就是 `acquire-load` 和 `release-store`
  * 原子变量默认情况下也是 `release-store` 和 `acquire-load` ？

```c++
// thread 1
acquire;
write x;
st.rel ready 1; // release store
read/write y; // 该条指令可以往前挪了，也可以保证 acquire-release 语义的正确性

// thread 2
read/write z; // 该条指令可以往后挪了，也可以保证 acquire-release 语义的正确性
ld.acq r0, ready; // acquire load
read x;
release;
```

### Sequential Consistence

* acquire-release 不保证全局序
  * 以下代码 可能 都被打印出来。
  * 原因是 `on-chip network` ： cpu 之间的消息传递有快有慢？？？？。

```c++
// x, y is std::atomic
// thread 1
x = 1;

// thread 2
y = 1;

// thread3
if (x == 1 && y == 0) {
    print("x first");
}

// thread 4

if (y == 1 && x == 0) {
    print("y first")
}
```

* Sequential Consistence
  * on-chip network保证消息传播的序，即先**传播 x=1**  到所有的处理器，等到所有的处理器都收到 x =1 之后，再传播 y = 1；反之亦然。总之：**写操作串行的使用片上网络**，从而片上网络这一层有了全局序。
  * 如果是 `std::atomic` 是 `sc` 那么上面代码永远不可能两个同时打印。



### Relaxed Memory Order

* x86上性能提升有限。不建议使用。

* C++ 几种 `memory-order` 总结
  * `memory_order_relaxed` : 用于 `load/store` 操作是原子的，但是没有顺序的保证 
  * `memory_order_release`： 用于 `store`，表示 `release` 语义
  * `memory_order_acquire` ：用于 `load`, 表示 `acquire` 语义
  * `memory_order_acq_rel` ：用于 `load` 或者 `store`, 对于 `store` 表示 `release`, 对于 `load` 表示 `acquire` 语义
  * `memory_order_seq_cst` ： 用于 `load/store`，表示顺序一致性
  * `memory_order_consume`：用于 `load`，表示 `consume` 语义。 （不建议使用！）
  * 其他的组合则是没有意义的。
* **线程启动** 和 **线程 join** 里面都是有 `memory-barrier` 的。

* `Relaxed Memory Order` 应用场景
  * **只保证操作的原子性**， 不保证 `memory-order`
  * 计数器
  * 简单标志。

### Singleton常见错误

```c++
// 错误示例
MyClass *get_instance() {
    if (p == nullptr) { // 在没有锁的情况下 访问了共享数据，导致 race
        mutex.lock();
        if (p == nullptr) {
            p = new MyClass;
        }
        mutex.unlock();
    }
}

// 原因详解, p = new Myclass 实际是拆成两部分实现的
p = malloc(sizeof(MyClass)); // 这个时候 p 已经不是 nullptr 了。
new(p) MyClass();
```







# 参考资料

[https://www.cnblogs.com/haippy/p/3237213.html](https://www.cnblogs.com/haippy/p/3237213.html)

[http://www.bogotobogo.com/cplusplus/multithreaded4_cplusplus11.php](http://www.bogotobogo.com/cplusplus/multithreaded4_cplusplus11.php)

[http://www.bogotobogo.com/cplusplus/C11/7_C11_Thread_Sharing_Memory.php](http://www.bogotobogo.com/cplusplus/C11/7_C11_Thread_Sharing_Memory.php)

[http://www.cplusplus.com/reference/condition_variable/condition_variable/](http://www.cplusplus.com/reference/condition_variable/condition_variable/)

[http://www.cplusplus.com/reference/mutex/unique_lock/](http://www.cplusplus.com/reference/mutex/unique_lock/)

[http://blog.csdn.net/liuxuejiang158blog/article/details/17263353](http://blog.csdn.net/liuxuejiang158blog/article/details/17263353)





