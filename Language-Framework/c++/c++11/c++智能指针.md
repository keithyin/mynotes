# c++智能指针

在c++中，动态内存分配（堆内存分配）主要由`new` 和 `delete` 两个操作符控制。`new`用来分配和初始化内存,`delete` 用来释放由`new` 分配的存储空间。

但是，使用这两个操作符可能会出现问题，

* 因为我们极有可能忘记`delete`，这就会导致内存泄漏。
* 还有另一个指针指向了该内存，但是被释放了。



为了可以更安全的使用动态内存，所以在`c++11`中增加了**智能指针**,用来**管理动态内存**, 它们分别是:

> ownership : 负责 对象的释放. 不具有的话就是不负责对象的释放

* shared_ptr (具有对象的 ownership)
* unique_ptr(具有对象的 ownership)
* weak_ptr (不具有对象的 ownership)

> g++中的在 boost 中 `boost::shared_ptr<class T>` 
>
> c++ primer 中说的 在 <memory> 头文件中
>
> 智能指针也就是指针，只不过比较智能而已（自己管理引用计数）

**限制**

* 只能管理用 `new` 创建的, 可以被 `delete` 的对象
* 保证对于 `managered object` 只能有一个 `manager object`
  * `manager object` 是使用 `raw pointer` 作为实参 构建 `shared_ptr` 时候构建的. 所以, 只要 `raw_pointer` 不被放进两个 `shared_pointer` 构造函数中即可 

## shared_ptr

* shared_ptr<class T> 也是个模板类

  ```c++
  shared_ptr<string> p1; // shared_ptr that can point at a string
  string* s_p1; //normal ptr
  shared_ptr<list<int> > p2; // shared_ptr that can point at a list of ints
  
  // 如何判断 shared_ptr 是否为空(还未被初始化) , 直接 if 中搞就行了.
  if (p1) {}
  ```

* `shared_ptr<T> p(q)`

  * `p` 是 `q(shared_ptr)`的一个副本，增加`q`中的计数。`q`中的指针必须可以转化为`T*`

* `p=q`

  * `p` 和 `q` 是 `shared_ptr`，保存的指针，类型之间可以互相转化。减小`p`中的引用计数，增加`q` 中的引用计数

* `p.unique()`

  * 如果引用计数为1，返回`true`

* `p.use_count()`

  * 返回`p`引用计数

* 通过`new`运算符初始化`shared_ptr`

  ```c++
  shared_ptr p2(new int(42)); //p2 指向一个 int值
  ```

* `make_shared<T>(args)` 函数

  * 返回一个智能指针，指向动态分配的对象（`T类型`）

    ```c++
    shared_ptr<int> p3 = make_shared<int>(42); //args用来初始化对象
    ```

* `shared_ptr` 会自动删除它们的对象（调用管理对象的析构函数）

  * 当指向某个对象的最后一个`shared_ptr`被销毁，那么这个对象也会被自动销毁。

```c++
//visual studio 2013 test demo
#include <memory>
#include <iostream>
using namespace std;

int main(){
	shared_ptr<int> p(new int(32));
	cout << "reference:" << p.use_count() << endl;
	cout << *p << endl;
	shared_ptr<int> p2(p);
	cout << "reference:" << p2.use_count() << "    reference :" << p.use_count() << endl;
	cout << *p2 << endl;
	*p2 = 2;
	cout << *p2 << endl;
	cout << *p << endl;
}
```

## week_ptr

* 仅仅观察 由  `shared_ptr` 管理的对象, 提供判断 对象是否 还存在的方法 (`week_ptr` 与 `shared_ptr` 一起使用)
  * 什么情况下 , `shared_ptr` 管理的对象没了?????
  * 相比与 `raw_pointer` 的优点是, `week_ptr` 知道它 `observe` 的对象到底还存不存在. `raw_pointer` 由于不知道存不存在, 直接访问的话, 执行的代码会 异常的.
* 只能由 `shared_ptr `初始化, 没有 `*, ->, get()` 方法, 只能转成 `shared_ptr` 之后操作
* `wp.lock()` 检查 观察的 `object` 还在不在, 如果在 返回一个 `shared_ptr`, 如果不在, 返回一个空的 `shared_ptr`

## unique_ptr

> 一个 unique_ptr "拥有"其指向的对象。与 shared_ptr 不同，**某个时刻**，只能有一个 unique_ptr 指向一个给定对象（shared_ptr 允许多个指针指向同一个对象）。当 unique_ptr 被销毁时，它所指向的对象也被销毁。

* 和 `raw_pointer` 相比, 零 `overhead` . 并没有像 `shared_ptr` 一样还分配了一个 `mamager object`
* 不支持赋值
* 不支持拷贝
* 支持移动拷贝

```c++
unique_ptr<double> p1; //可以指向double 的 unique_ptr
unique_ptr<double> p2(new double(42.0)); // p2指向一个值为42.0 的double
unique_ptr<double> p3(p2); //错误，unique_ptr 不支持拷贝
p1 = p2; //错误：unique_ptr 不支持赋值

p2.release() ; // p2放弃对指针的控制权，返回指针
```

## make_shared

```c++
shared_ptr<Demo> p(new Demo); // 会两个分配内存, new Demo一次, 创建 manager object 一次
shared_ptr<Demo> p(make_shared<Demo>());// 只会分配一次内存, 放 new Demo 和 manager object
```



## enable_shared_from_this

[https://en.cppreference.com/w/cpp/memory/enable_shared_from_this](https://en.cppreference.com/w/cpp/memory/enable_shared_from_this)

[https://stackoverflow.com/questions/712279/what-is-the-usefulness-of-enable-shared-from-this](https://stackoverflow.com/questions/712279/what-is-the-usefulness-of-enable-shared-from-this)

* 功能：使得 指向对象 `t` 的智能指针能够调用 `shared_from_this()` 构建更多的智能指针
  * 使得 同一个 `managered object` 只有一个 `manager object`
* 如果不是 `t` 的智能指针调用 `shared_from_this()` ，会报错。
* 应用场景:  
    * 类方法中 调用其它方法, 如下面 Good 类中 JustADemo 调用了SomeFunc, SomeFunc 的形参是一个 shared_ptr, 实参是当前的对象(this). 如果 调用的对象本身是一个 shared_ptr, 如果直接将 this 传给 SomeFunc, 这就导致 this 被 两个 manager object 管理, 会导致 duoble free 的问题. 
    * 这时候使用 `enable_shared_from_this` , 然后调用 `shared_from_this()` 得到 其 `shared_ptr` , 然后就可以随便传递咯. 这个 shared_ptr 和本身的就是同一个 manager object 了.
    * 这里需要注意的是, 如果不是 shared_ptr 取调用 带有 `shared_from_this()` 方法时, 会报错. 

```c++
#include <memory>
#include <iostream>

struct Good: std::enable_shared_from_this<Good> // note: public inheritance
{
  // 返回 this 指针的 shared_ptr, 共用此前的 manager object
    std::shared_ptr<Good> getptr() {
        return shared_from_this();
    }
    
    void JustADemo() {
        SomeFunc(shared_from_this());
    }
};

void SomeFunc(std::shared_ptr<Good> p) {
    // do some thing
}
 
struct Bad
{
    std::shared_ptr<Bad> getptr() {
        return std::shared_ptr<Bad>(this);
    }
    ~Bad() { std::cout << "Bad::~Bad() called\n"; }
};
 
int main()
{
    // Good: the two shared_ptr's share the same object
    std::shared_ptr<Good> gp1 = std::make_shared<Good>();
    std::shared_ptr<Good> gp2 = gp1->getptr();
    std::cout << "gp2.use_count() = " << gp2.use_count() << '\n';
 
    // Bad: shared_from_this is called without having std::shared_ptr owning the caller 
    try {
        Good not_so_good;
        std::shared_ptr<Good> gp1 = not_so_good.getptr();
    } catch(std::bad_weak_ptr& e) {
        // undefined behavior (until C++17) and std::bad_weak_ptr thrown (since C++17)
        std::cout << e.what() << '\n';    
    }
 
    std::shared_ptr<Good> bp1 = std::make_shared<Good>();
    // shared_from_this, 私有成员函数
    std::shared_ptr<Good> bp2 = bp1->shared_from_this();
    std::cout << "bp2.use_count() = " << bp2.use_count() << '\n';
} // UB: double-delete of Bad
```



# 如何正确的使用 智能指针

https://www.modernescpp.com/index.php/c-core-guidelines-rules-to-smart-pointers

https://stackoverflow.com/questions/2454214/is-it-a-good-practice-to-always-use-smart-pointers



## 参考资料

http://www.umich.edu/~eecs381/handouts/C++11_smart_ptrs.pdf
