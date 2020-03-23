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

