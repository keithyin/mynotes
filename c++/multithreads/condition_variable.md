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



