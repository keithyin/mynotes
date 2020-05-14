# 第一章: 关于对象

* `C++` 中,  数据成员 和 函数成员
  * 两类 `class member datas : `static`  和 `non-static`
  * 三类 `class member functions` : `static` , `non-static`, `virtual`

## 几种常见对象模型

* 简单对象模型 (`a simple object model`)
  * `class object` 是 一系列的 `slots` , 每一个 `slot` 都指向一个 `member` . `Members` 按照声明的顺序, 各被指定一个 `slot`.  `data members` 和 `function members` 都有自己的 slot.
  * `slots` 中存的是指针.
  * `members` 本身并不放在 `object` 中, 只有指向这些 `member` 的指针放在 `object` 中.
  * 优缺点:
    * 优点: `object` 的大小容易计算. 避免 `member` 有不同的类型, 因而需要不同的存储空间所招致的问题? 招致啥问题??
* 表格驱动对象模型 (`a table-driven object model`)
  * 将所有与 `members` 相关的信息抽取出来, 放在 `member data table` 和 `member function table` 中. `class object` 只包含指向这两个表格的指针.
  * `member data table` 直接用于数据本身
  * `member function table` 则是一系列的 `slots` , 指向对应的函数.



## C++ 对象模型

`C++ ` 对象模型

* 数据成员
  * 非静态数据成员: 放在每个 object 之内
  * 静态数据成员: 放在 object 之外
* 函数成员:
  * 静态的 和 非静态 的也是放在 object 之外
  * `virtual` 成员函数 
    * 每一个`class` 产生一堆指向 `virtual functions` 的指针, 这些指针放在一个表格之中. 这个表格被称为 `virtual table`
    * 每一个 `class object` 被安插一个指针, 指向相关的 `virtual table` . 通常这个指针称之为 `vptr`.  `vptr` 的 `set` 和 `reset` 都由每一个 `class` 的 构造函数, 析构函数, 赋值构造函数 来完成. 每一个 `class` 所关联的 `type_info` 对象 也经由 `virtual table` 被指出来, 通常放在 表格的 第一个 `slot`中.
      * 多继承怎么办????? 多个 virtual table?  多个 vptr ? 



#### 小总结

* `virtual table` 是 每个 类一个, 而不是 每个对象一个!!



## 继承

* `derived class` 如何塑造 `base class` 的实例
  * `C++` 最初采用的方式 直接将 `base class subobject` 的 `member data` 直接放在 `derived class object` 中.
  * `C++2.0` 使用 了 `virtual base class`



### 存储一个对象需要多大的内存空间

* `nonstatic data members` 总和大小
* 加上需要 `aligment` 的需求而填补的空间. 可能存在于 members 之间, 也可能存在于集合的边界. `aligment` 通常是将内存的数值调整到某数的倍数. 在 32 位的计算机上, 通常 aligment 位 4bytes(32bits), 已使 bus 的运输量达到最高效率.
* 加上为了支持 `virtual` 机制而由内部产生的额外负担.



### 对于指针的理解

* 指针的**类型** 会教导编译器如何解释某个特定地址中的内存内容及大小
* `void*` 指针只有一个地址, 而不能通过它操作所指的 `object`, 因为我们不知道 `void*` 指针涵盖多少空间
* `cast` 其实是一种编译指令, 大部分情况下, 它并不会改变一个指针所含的真正地址, 它只影响 `被指出之内存的大小和其内容` 的解释方式. 



```c++
class Bear: public ZooAnimal {
public:
  void rotate(){}
 private:
  int cell_block_;
};
Bear bear;
Bear* pb = &bear;
ZooAnimal* pa = &bear;

ZooAnimal za = bear; // 这会引起切割, 同样也不会触发虚机制

grep "cmatch=(669" feedas.log.20200331* | grep "47920-dz" | awk -F'uc_freq' '{print $2}' | awk -F'|' '{++pv; winfo+=$2; user+=$3; title+=$6;}END{print pv,winfo/pv,user/pv,title/pv}'

```



# 第二章: 构造函数语义学

> 继承, 多继承, 虚机制, 多继承的虚机制
>
> 构造函数: 构造函数, 拷贝构造函数, 移动构造函数, 析构函数

* `explicit` : 防止单一参数的 `constructor` 被当做一个 `conversion` 运算符
* `memberwise initialization?`
* `named return value?`

## Default Constructor

* 默认构造函数 在需要的时候被 编译器生产出来
  * 谁需要: 编译器需要的时候
  * 搞出来做什么事情: 
* 这里要分清: 编译器的责任 和 程序员的责任.

```c++
class Foo {public: int val;};
void foo_bar() {
	Foo bar; // 程序要求 bar's members 都被清0
}
```



* nontrivial default constructor 的四种情况
  * **类中有**  带有 `defalult constructor` 的 `member class object` 
    * (这个default contructor可以是implicit生成的?)
    * 在 真正被调用的时候才会被 合成构造函数
    * 如果在真正调用的时候合成, 如何解决链接时候 函数重复实现的问题呢? 使用 inline 的方式. (`default (constructor, copy constructor, destructor, assignment copy operator) ` 都用 inline 的方式.)
    * 这时候编译器合成出来`default contructor` 的作用是: 负责调用 `member class object` 的 `default contructor` 
  * **类中有** 带有 `default constructor` 的 `base class`
    * 作用: 用来调用 `base class` 的 `defalult constructor` , 如果是多继承, 调用的顺序是 声明的顺序
  * 带有 `Virtual Function` 的 `class`
    * 作用: 为了正确的设置 `vptr` 的值, 使虚机制正常工作.
  * 带有一个 `virtual base class` 的 `class` (虚继承?)
* 如果自定义了 `constructor` : 会在自定义的代码中插入 对于 `father ....` 的一些构造操作.



## Copy Constructor 的构造操作

> C++标准同样将 copy constructor 分为两种, 一种是 trivial, 一种是 non-trivial
>
> * 只有 non-trivial 的才会自动生成. trivial 的根本不会合成出来.
> * 决定是否是 trivial 的, 是 class 是否表现出 bitwise copy sementics
>
> 
>
> 有三种情况, 会以一个`object` 的内容作为另一个`class object` 的初值
>
> * 直接赋值, 参数传递, 返回值.

```c++
class X {};
void foo(X x) {}
X foo_bar(){ return X;}
X x;
X xx = x; // 1. 直接赋值

foo(x); // 2. 对象作为参数传给函数
X xxx = foo_bar(); // 3. 函数返回一个 class object 时.
```

* 自定义`Copy Constructor`

```c++
X::X(const X &x){}
Y::Y(const Y &y, int i=0){} // 可以是多个参数的形式, 之后的参数需要提供一个默认值
```

* 如果用户没有自定义, 那么编译器将会自动生成一个 `memberwise initialization` 的复制构造函数
* 如果class是 `bitwise copy semantics` , 那么编译器是不会合成 `copy constructor` 的.

```c++
class Word {
  int cnt;
  char* str;
}; // 这种情况下是不需要合成一个 default 的copy constructor 的, 因为 bitwise copy 足够

class Word2 {
	int cnt;
  string str;
};// 这时候编译器为了正确的 调用str 的copy constructor, 所以需要给 Word2 合成一个copy constructor

```

* 什么时候 一个`class` 不展现出 `bitwise copy semantics` 呢?
  * `class` 内含一个 `member object` , 且 后者的 `class` 声明了一个 `copy constructor(编译器合成的, 自己定义的都可以)` 
  * `class` 继承一个 `base class` , 且后者存在一个 `copy constructor(不论是显示声明 还是 合成)` 时
  * 当 `class` 声明了一个或多个 `virtual functions` 时
  * `class` 继承链上有 `virtual functions` 时
* 为什么 `virtual functions` 要特殊对待呢? 有 `virtual functions` 的类 的对象只是多了一个 `vptr` 而已. 这个东西的复制构造 考虑一下可能会存在什么样的问题.

```c++
class Father{}; // 内部会有一些 虚函数
class Son: Father{};

void draw(const Father& obj){obj.draw();}

void main(){
  Son son;
  /* father 的 vptr 不应该指向 Son 的 virtual table, 但是如果是 bitwise 的复制的话就会发生这种情况
  所以说 Father 应该被合成出来 copy constructor, 用来正确的设置 自身的 vptr
	*/
	Father father = son; //这里赋值的话会发生切割行为
  draw(son);
  draw(father); 
}
```

* `virtual base class` 如何处理?

