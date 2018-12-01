# c++智能指针

在c++中，动态内存分配（堆内存分配）主要由`new` 和 `delete` 两个操作符控制。`new`用来分配和初始化内存,`delete` 用来释放由`new` 分配的存储空间。

但是，使用这两个操作符可能会出现问题，

* 因为我们极有可能忘记`delete`，这就会导致内存泄漏。
* 还有另一个指针指向了该内存，但是被释放了。



为了可以更安全的使用动态内存，所以在`c++11`中增加了**智能指针**,用来**管理动态内存**, 它们分别是:

* shared_ptr
* unique_ptr
* weak_ptr

> g++中的在 boost 中 `boost::shared_ptr<class T>` 
>
> c++ primer 中说的 在 <memory> 头文件中
>
> 智能指针也就是指针，只不过比较智能而已



## shared_ptr

* shared_ptr<class T> 也是个模板类

  ```c++
  shared_ptr<string> p1; // shared_ptr that can point at a string
  string* s_p1; //normal ptr
  shared_ptr<list<int> > p2; // shared_ptr that can point at a list of ints
  ```

* shared_ptr<T> p(q)

  * `p` 是 `q(shared_ptr)`的一个副本，增加`q`中的计数。`q`中的指针必须可以转化为`T*`

* p=q

  * `p` 和 `q` 是 `shared_ptr`，保存的指针，类型之间可以互相转化。减小`p`中的引用计数，增加`q` 中的引用计数

* p.unique()

  * 返回`true`，如果引用计数为1

* p.use_count()

  * 返回`p`引用计数

* 通过`new`运算符初始化`shared_ptr`

  ```c++
  shared_ptr p2(new int(42)); //p2 指向一个 int值
  ```

* make_shared<T>(args) 函数

  * 返回一个智能指针，指向动态分配的对象（`T类型`）

    ```c++
    shared_ptr<int> p3 = make_shared<int>(42); //args用来初始化对象
    ```

* shared_ptr 会自动删除它们的对象

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



## unique_ptr

> 一个 unique_ptr "拥有"其指向的对象。与 shared_ptr 不同，某个时刻，只能有一个 unique_ptr 指向一个给定对象（shared_ptr 允许多个指针指向同一个对象）。当 unique_ptr 被销毁时，它所指向的对象也被销毁。

```c++
unique_ptr<double> p1; //可以指向double 的 unique_ptr
unique_ptr<double> p2(new double(42.0)); // p2指向一个值为42.0 的double
unique_ptr<double> p3(p2); //错误，unique_ptr 不支持拷贝
p1 = p2; //错误：unique_ptr 不支持赋值

p2.release() ; // p2放弃对指针的控制权，返回指针
```

