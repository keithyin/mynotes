# Effective C++ 读书笔记

## 约定

* 构造函数尽量声明成 `explicit` ，防止编译器进行隐式的类型转换
* `copy` 构造函数的语义是：以同类型对象初始化自我对象
* `copy assignment` 的语义是：从另一个同类型对象中拷贝其值到自我对象



## 条款

* 尽量使用 `const, enum, inline` 替代 `#define` 
  * 原因：使用 `#define` 不好调试，因为 `#define` 在预处理阶段就会消失。
* 尽最大 可能的使用 `const` 。
  * 常量性不同的函数可以被重载
* 确定对象在被使用前已先被初始化
* 了解 `C++` 默默编写并调用哪些函数
  * 构造函数，copy构造函数，copy assignment，析构函数
* 若不想使用编译器自动生成的函数，就该明确拒绝
  * `private` 读取权限 且 只声明不实现
  * `c++11` 似乎有更好的方法
* 为多态基类声明 `virtual` 析构函数
  * 保证资源释放的时候要正确
* 别让异常逃离 析构函数
  * 析构函数中出现异常要就地解决，千万不要在析构函数中抛出异常。
* 绝对不要在构造，析构的时候调用虚拟函数
  * 这个特性和 java/c# 不一样
  * 构造析构中调用 `virtual` 函数不会导致多态！！！！！
* 另 `operator=` 返回一个 `reference to *this`
  * 目的是为了赋值的连锁形式，不用 `reference to *this` 应该也可以正常连锁吧。
    * `(x=y)=15;` 这种操作还是需要 `reference to *this` 的。！！！！所以还是按照推荐返回就好了。
  * 赋值采用的是右结合律 `x=(y=(z=5))`
* 在 `operator=` 中处理 **自我赋值**
  * 自我赋值，自己赋值给自己
  * 如果是自我赋值，就什么事都不干
  * `if (this==&rhs) return *this;`
* **复制对象的时候一个成分都不能丢**
  * 复制构造函数和 `copy assignment` 函数不能互相调用，从语义上就解释不通，如果两个函数有相似的代码，那就再封装一个私有方法供两个函数调用。





**资源管理**

-----

> 资源：内存，文件描述符，互斥锁，sockets，等等

* 以对象管理资源
  * 思想：将资源放在对象中，利用对象的析构函数的自动调用机制来释放资源
  * 所以，指针最好存放在 **对象中**, `delete` 语句最好写在析构函数中。
  * 如何做：
    * 获得资源后，立刻放进 **资源管理对象** 中
    * 资源管理对象运用 析构函数确保资源被正确释放
    * `c++` 中的智能指针 其实就是**资源管理类**。
  * 总之：资源释放还是交给 对象的析构函数释放比较放心。
  * 函数返回值最好也不是指针，应当是智能指针。
  * 智能指针仅作用于 `head-based` 资源上。
* 在资源管理类中注意 `copying` 行为，**资源管理类**有以下几种行为可以选择
  * 禁止复制，将 `copying` 函数设置为 `private`
  * 对底层资源祭出 引用计数法，这个可以使用 `shared_ptr` 帮忙
  * 复制底部资源，即执行深度拷贝
  * 转移底部资源的拥有权
* 在资源管理器中提供对原始资源的访问
  * 因为有些 `API` 就直接开搞原始资源
  * 希望你写的代码不直接开搞原始资源
* 成对使用 `new` 和 `delete` 的时候要采取相同的形式
  * `new a` & `delete a`
  * `new a[100]` & `delete[] a`; ,`delete[] `用来表示 `a` 是数组第一个元素的地址
* 以 **独立语句** 将 `new` 出来的对象放到智能指针中
  * `shared_ptr<Widget> pw(new Widget);` 这么写是最好的。




**设计与声明**

----

* 让接口容易被正确使用，不易被误用
  * 接口设计保证一致性原则，即：语义要一样
* 设计 `class` 犹如设计 `type`
  * 自然的语法，直观的语义
  * 需要考虑以下几点
    * 新 `type` 对象应该 如何创建和销毁
      * 这会影响到 `operator new, operator new[], operator delete, operator delete[]` 方法的编写。
    * 对象的 初始化 和 对象的赋值应该有什么区别
    * 新 `type` 对象如果被 `pass by value` 应该发生什么
      * `copy constructor` 用来定义一个 `type` 的 `pass by value` 如何实现
    * 什么是新 `type` 的合法值
    * 新 `type` 需要配合某个继承图系吗？
    * 新 `type` 需要什么样的类型转换
* 宁以 `pass-by-reference-to-const` 代替 `pass-by-value`
  * 可以防止调用复制构造函数，高效而且可以避免切割问题
  * 当然以下类型 使用 `pass-by-value` 也可
    * 内置基础类型
    * stl 迭代器
    * 函数对象
* 必须返回对象时，千万不要返回其 `reference` 。
  * 绝对不要返回一个 `pointer` 或 `reference` 指向一个 `local` 对象
  * 指向 `local` 对象的`pointer` 最好是通过智能指针返回出来。
* 将成员变量声明为 `private`， 不要声名成 `protected`和 `public` 哦。
  * 能够更好的控制读写访问
  * 如果成员变量改了，函数方法还可以保持不变
* 宁以 `non-member, non-friend` 替换 `member` 函数
  * 见 `术语--> 封装`
* 若 **一个函数的所有参数** 皆需要类型转换，请为此采用 `non-member` 函数
  * 因为交换率可能会满足不了
  * 重载运算符经常会有这种感觉
* 考虑写出一个不抛异常的 `swap` 函数



**实现：**

----

* **尽可能延后**变量定义式的出现时间

  * 创建对象而不使用是非常浪费资源的（时间，内存）
  * 对象最好是**创建的时候就进行初始化**，这样比创建后赋值要有效率
  * for 循环的时候还是在循环体里面定义比较靠谱

* 尽量少做转型动作

* 避免返回 `handles` 指向对象内部成分

  * 这样可以通过返回的 `handle` 来操作对象的内部成分，**这种做法是非常不安全的**。

* 为异常安全而努力是值得的

  * 异常安全性，（当异常被抛出时，应该满足以下两个条件）

    * 不泄露任何资源，（即使抛异常，该回收的资源也要回收）：
      * **资源管理类可以协助做好这个工作。**
    * 不允许数据败坏（指针指向垃圾数据）

  * 异常安全函数提供以下 **三个保证之一**, （**写代码的时候一定要满足一个**）

    * 基本承诺
      * 如果异常被抛出，程序内的任何事物仍然保持在 **有效状态下**
    * 强烈保证
      * 如果函数成功，则是完全成功，如果函数失败，程序会恢复到调用函数之前的状态
      * `copy and swap` 可以协助做好这个工作， 要么全有，要么全无。
    * `no throw` 保证
      * 承诺绝不抛出异常，因为它们总能完成他们原先承诺的功能

  * 透彻了解 `inlining` 的里里外外

    * 好处：免除函数调用成本
    * 坏处：增加软件体积

  * 将文件间的编译依存关系降到最低

    * 将一个类划分成两个 `classes` 一个只提供接口，一个负责实现，然后对 实现的那个类进行前置声明就好了。

    * 编译器必须在  **编译器间知道对象的大小** 

      * 返回指类型有没有定义过没有关系，只需要声明就够了

      ```c++
      // some.h
      class Date;

      /*以下两个声明都不需要 Date 的定义式*/
      Date tody();
      void clearAppointments(Date d); 
      ```

      ​

    * 尽量让**头文件自我满足** ，如果做不到，则让它与其它文件中的声明式（而非依赖式）相依！！

    * 使用 声明式 减少 `#include` 语句。

    * 尽量遵守以下两个规则

      * 如果能使用 `object references` 或 `object pointers ` 完成任务就不要使用 `object`
      * 如果能够，尽量以 `class 声明式` 代替  `class定义式`

  ```c++
  #include <string>
  #include <memory>

  class PersonImpl;
  class Date;
  class Address;
  class Person{
    public:
    	Person(const std::string& name, 
             const Date& birthday, const Address& addr);
    	std::string name();
    	std::string BirthDate();
    	std::string Address();
    private:
    	std::shared_ptr<PersonImpl> impl; //指针，指向实现物
  }
  /*
  这样一搞，就不用 #include <personimpl> 这些东西了。
  这样，包含此头文件的 .cpp 就不会因为 PersonImpl，Date，Address 头文件的改变而需要重新编译了。
  */
  ```





**继承与面向对象设计**

-----

>public 继承意味着 is-a
>
>virtual 函数意味着： 接口必须被继承
>
>non-virtual 函数意味着：接口和实现都必须被继承

* 确保你的 `public` 继承塑造出 `is-a` 关系， (每一个 D 都是 B)
  * 好的接口可以防止无效的代码通过编译
  * 什么是 `is-a` 关系，并不是数学中的概念
    * 能够在 `base class` 对象身上做的每件事情，都可以在 `derived-class` 对象身上做
* 避免遮掩继承而来的名称，虽然遮掩依旧可以访问，但还是不要这么做
* 区分接口继承和实现继承
  * 接口继承：只继承成员函数的声明。
  * 实现继承：同时继承函数的声明和实现
* 考虑 `public virtual` 函数以外的其它选择
  * `public virtual` 提供了多态，多态也可以由多种方式提供
  * 使用 `NVI` （见术语），这就相当于 给 virtual 成员函数加了个 wrapper，可以做一些额外的工作
  * 考虑 策略模式，将算法封装起来。这样会比采用 `virtual` 成员函数提供更大的灵活性
* **绝不重新定义** 继承而来的 `non-virtual` 成员函数
  * 如果想重新定义的话，搞个 `virtual` 吧
* **绝不重新定义** 继承而来的 **缺省参数值** (函数的默认参数值)
* 通过组合建模出 `has-a` 或 `is implemented in terms of`
* 明智而审慎的使用 `private` 继承
  * 编译器不会自动将一个 `derived class` 对象转化成 `based class` 对象
* 明智而审慎的使用多重继承



**模板与泛型编程**

-----

* 了解隐式接口和编译期多态
  * 模板就提供了编译期多态
  * 显式接口由函数的签名式构成
  * 隐式接口由一组有效表达式构成
    * 有效表达式定义了 一组约束条件
* 了解 `typename` 的双重意义
  * 用来说明嵌套从属类是个类
  * `typename T::iterator iter;` 如果不用 `typename`，编译器将不将 `iterator` 解析成为类。
  * 但是不可以在 `base class list` 和 `member initialization list` 中使用。
* 学习处理模板化基类内的名称, (**考虑的是模板继承中的问题**)

```c++
template <typename T>
class B{
public:
  void do(){
    //...
  }  	
  
};

template <typename T>
class D: public B<T>{
public:
  void haha(){
    do(); // 这里会报编译异常
  }
};
/*
C++ 模板支持特例化，所以无法确定 `B<T>` 的某个特例化版本是否含有 do() 这个函数，
因此 C++ 选择报错。（编译器的两个阶段）

有三种方法可以假设基类是有某些函数的, 让编译器放松警惕
	this->do();
	using B<T>::do; do()
	B<T>::do();
*/
```



* 将与 **模板参数无关** 的代码抽离 `template`
  * 避免代码膨胀
* 运用 **成员函数模板** 接收所有兼容类型
* 需要类型转换时 请为模板定义非成员函数
* 请使用 `traits classes`  表现类型信息



**定制 new 和 delete**

----

> `new-handler` 这是当 `operator new` 无法满足客户的内存需求时所调用的函数
>
> `operator new` 和 `operator delete` 只适合用来分配单一对象
>
> `operator new[]` 和 `operator delete[]` 是用来操作 `Array` 的

* 了解 `new-handler` 的行为

  * 当 `operator-new` 无法满足某一内存分配需求时，它会抛出异常，在抛出异常前，它会先调用一个客户指定的错误处理函数，一个所谓的 `new-handler`。 为了指定这个 `用以处理内存不足` 的函数，客户必须要调用 `set_new_handler` ，那是声明于 `<new>` 的一个标准库函数

    ```c++
    // new
    namespace std{
      typedef void (*new_handler)();
      new_handler set_new_handler(new_handler p) noexcept;
    }
    ```

    ```c++
    // demo.cc
    void OutOfMem(){
      std::cerr<<"unable to satisfy request for memory"<<std::endl;
      std::abort();
    }

    int main(){
      std::set_new_handler(OutOfMem); //设置全局的
      int* p = new int[10000000];
    }
    ```

  * 一个设计良好的 `new-handler`  必须要做以下事情

    * 让更多内存可被使用
    * 安装另一个 `new-handler`
    * 卸除 `new-handler`
    * 抛出 `bad_alloc`（或派生自 `bad_alloc`） 的异常
    * 不返回。（通常调用 `abort` 或 `exit`）

* 了解 `new`  和 `delete` 的合理替换时机

  * 为什么要替换掉编译器提供的 `operator new` 和 `operator delete` 呢？
    * 用来检测运用上的错误：
    * 为了强化性能：编译器提供的默认 `new/delete` 为了满足大部分应用场景，显的比较臃肿，所以自定义 `new/delete` 可能会对当前的程序有帮助
    * 为了收集使用上的统计数据

* 编写 `new/delete` 时需要墨守成规

  * ​






## 术语

* `RAII` : Resource Acquisition Is Initialization, 在构造函数中获取资源，在析构函数中释放资源
* `封装`： 如果有些东西被封装，他就不再 **直接** 可见，越多的东西被封装，就会有越少的人可以看到他，而越少人看到他，我们就有越大的弹性去改变他，因为我们的改变仅仅影响那些可以直接看到他的事物。所以对于私有成员来说，越少的函数可以直接访问他是非常好的。
* `NVI`： Non Virtual Interface，用户使用 `public non-virtual`成员函数调用 `private virtual` 函数。也叫做 模板方法 设计模式。
* (enum hacks)[https://www.linuxtopia.org/online_books/programming_books/thinking_in_c++/Chapter08_023.html]: 
  * 匿名 `enum`
  * 编译器确定 `enum` 元素的值
  * 不占用对象的存储空间
* ​