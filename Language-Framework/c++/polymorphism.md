# 多态性与虚函数
## 向上类型转换
对象 可以作为 自己的类 或者 作为它的基类的对象 来使用。

## 早捆绑与晚捆绑
把函数体 与 函数调用相联系称之为捆绑。 当捆绑在程序运行之前，称之为早捆绑。当捆绑在程序运行时，称之为晚捆绑。

为了引起晚捆绑，c++要求基类中声明这个函数时使用`virtual`关键字，晚捆绑只对 `virtual` 函数起作用，而且只在使用 含有 `virtual` 函数的基类的地址时发生，尽管它们可以在更早的基类中定义。
## 虚函数
语法：基类成员函数前加个 `virtual` 即可，对于虚函数，只需要在声明的时候说明一下，定义的时候并不需要。

**注意:仅需要在基类中声明一个函数为虚函数，那么这个基类的子类中的和基类虚函数匹配的函数都将使用虚机制。**

```c++
class Instrument{
public:
  virtual void play() const{
    cout<<"Instrument play()"<<endl;
  }
}

class Wind: public Instrument{
public:
  //override 前面不需要加 virtual 了
  void play() const{
    cout<<"Wind play()"<<endl;
  }
}
```
虚机制使用虚表实现：
含有虚函数的类都有一个虚表，对象里有一个虚指针，指向这个虚表。



# RTTI

> 运行时类型识别

* 提供两个操作符
  * `dynamic_cast`
    * 安全的转成 `base_type` 或这 `derived type` , 转不过去就 `nullptr` 了
  * `typeid`
    * 如果没法搞 虚函数 , 那就只能用 `typeid` 了,  但是这种方法比较容易出错(`error-prone`)
    * The typeid operator allows a program to ask of an expression: `What type is your object?`

```c++
Base *bp = dp; // both pointers point to a Derived object
// compare the type of two objects at run time
if (typeid(*bp) == typeid(*dp)) {
// bp and dp point to objects of the same type
}
// test whether the run-time type is a specific type
if (typeid(*bp) == typeid(Derived)) {
// bp actually points to a Derived
}

// test always fails: the type of bp is pointer to Base
if (typeid(bp) == typeid(Derived)) {
// code never executed
}
```



