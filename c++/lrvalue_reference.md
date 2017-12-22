# C++ 中的右值与左值

## 右值与左值

* 右值：**只能出现** 在 等式 右边的值。
* 左值：**可以出现** 在 等式左边的值。



## 右值引用 与 左值引用

* 左值引用

```c++
int i = 0;
int &j = i;
```

上面代码中， `i` 是一个左值，因为 它既可以在等式的右边，也可以在等式的左边。`int &j=i;` 正确的表述应为：获得 `i` 的 左值引用。



* 右值引用：（获得右值的引用，只能绑定到将要销毁的对象。即：右值引用只能绑定到 右值上。）右值引用窃取了即将被销毁的对象的状态。

```c++
int i = 10;
int &&j = i*100;
```

对于一个有返回值的表达式来说，如果这个表达式的返回值没有对象来接收，他会创建一个临时对象，这个对象在语句结束后就会被销毁。所以 `i*100` 返回一个 右值。 所以 `int &&j = i*100;` 就是 `j` 获得了 `i*100` 的右值引用。



```c++
int i = 42;
int &r = i;
int &&rr = i; // 错误：i 是左值
int &r2 = i*42; // 错误，i*42 是右值
const int & r3 = i*42; // 正确，可以将 const 引用绑定到右值上
int &&rr2 = i*42; // 正确 ， i*42 是右值
```







## move 函数

虽然不能将 右值引用直接绑定到 一个 左值上，但是我们可以显示的将一个 左值转换成对应的 右值引用类型。

```c++
int ii = 111;
// 虽然我们有一个左值，但是我们希望像右值一样对待它
int && j = std::move(ii);
// 上面 这个 操作之后，除了对 ii 赋值 或者 销毁 它， 我们将不再用它。
```



## 移动构造函数 和 移动赋值运算符

```c++
StrVec::StrVec(StrVec&& s) noexcept // 移动操作不应该抛出任何异常
  : elements(s.elements), first_free(s.first_free), cap(s.cap){
    // 顶 s 进入这样的状态--对其运行析构函数是安全的
    s.elements=s.first_free=s.cap=nullptr;
  }
```

```c++
StrVec &StrVec::operator=(StrVec &&rhs) noexcept
  {
    if (this!=&rhs){
      free(); //释放已有元素
      elemets = rhs.elements; // 从 rhs 中接管资源。
      ...;
      ...;
      // 将 rhs 处于 析构状态。
      rhs.elements=rhs.first_free=rhs.cap=nullptr;
    }
  }
```

