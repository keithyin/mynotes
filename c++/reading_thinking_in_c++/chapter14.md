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

341

