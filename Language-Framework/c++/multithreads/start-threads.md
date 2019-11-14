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

**Advantage**

* 线程之间通信的最快方式



**RACE**





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







