# chapter 14

**继承与组合**



## 组合

类中有个其他类的对象作为成员



## 继承

```c++
class A : public Base{
  
};

class B : public Base1, public Base2{
  
};
```

继承：意味着 **子类**(A) 将 **包含父类**(Base) 中的**所有** 数据成员和成员函数。

实际上：正如没有对 Base 进行继承，而在 A 中创建了一个 Base 的成员对象一样，A 中包含了 Base 的一个子对象。**无论是成员对象还是基类存储，都被认为是子对象**。



### 构造函数的初始化列表

* 当创建一个**对象时**， 编译器确保调用了所有子对象的 构造函数
  * 子对象中如果有默认函数，编译器直接调用默认的构造函数就可以了
* 如果没有默认构造函数呢？ C++ 提供了解决方案
  * 构造函数初始化列表（子对象用来调用构造函数的地方）
  * 构造函数初始化列表允许我们显示的调用成员函数的构造函数。
  * 它的主要思想是：**在进入新对象的 构造函数体之前调用所有其它的构造函数**

```c++
class Base{
  int i;
  public:
  Base(int i){
    this->i = i;
  }
  void print(){
    cout << "hello Base" <<endl;
  }
}
class A : public Base{
  int m;
  public:
  A(int i):Base(i), m(i+1){
    // 用组合的话， 给出对象的名字就可以了
  }
  
}
```



### 成员函数重写 override

```c++
class Base{
  int i;
  public:
  Base(int i){
    this->i = i;
  }
  void print(){
    cout << "hello Base" <<endl;
  }
}
class A : public Base{
  int m;
  public:
  A(int i):Base(i), m(i+1){
    // 用组合的话， 给出对象的名字就可以了
  }
  void print(){
    Base::print(); // 重写了之后，用父类的名字，就只能这样咯
  }
}
```



## 三种继承方式

* public
* protected
* private



**无论什么继承，基类的私有变量对 子类都是不可见的**

**private继承**

* 共有变 私有

**protected 继承**

* 不常用，不用管

**public继承**

* public 还是 public



## 什么是没有继承的？

* 赋值运算符
* 构造函数
* 析构函数



## 向上类型转换

将子类的引用或指针，转变成 基类的引用或指针的活动被称为 **向上类型转换。**



**任何向上类型转换都会损失对象的类型信息**

```c++
Wind w;
Instrument * ip = &w;
```

编译器只能将 **ip** 作为一个 `Instrument` 指针来处理，它无法知道 `ip` 实际指向的是 `Wind` 对象。



## 组合还是继承

* 是否需要向上类型转换



## 总结

* 在进入 新对象的构造函数，所有的成员对象都会被正确的初始化。
* 如果有基类，基类先正确初始化，然后才是成员对象。

