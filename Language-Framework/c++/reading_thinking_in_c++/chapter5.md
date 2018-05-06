# chapter 5

Thinking in C++ 第五章阅读笔记



## C++ 访问控制

* private ： 只有内部成员函数可以访问。
* protected：只有内部成员函数可以访问，和 private 的区别在 继承情况下
* public：公有，谁都可以访问
* 友元（friend）： 哥俩好，哥俩好



**关于友元：** 必须在当前类中 声明谁是你的友元，在友元函数中，可以直接访问 对象的私有属性

```c++
class X;
class Y{
  public:
  void f(X*);
}
class X{
  private:
  int i;
  public:
  friend void g(X*, int); // 传X* 是非常有必要的，因为现在 X 的定义还不完整
  friend void Y::f(X*);
  friend class Z; // 友元类
  friend void h();
}

void X::g(X* x, int i){
  x->i = i; //因为是友元，所以才可以直接访问私有属性
}

void Y::f(X* x){
  x->i = 47; // 因为是友元
}

class Z{
  private:
  int j;
  public:
  void g(X* x);
}
void Z::g(X* x){
  x->i = j; // 因为是友元类，友元类里面的所有成员函数都可以直接访问 X 的私有变量
}

```



**注意：嵌套的 类 不能自动获得访问 private 成员的权限**



## 继承情况下的类成员的访问属性的变化



