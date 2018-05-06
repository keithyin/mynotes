# constexpr 与常量表达式

**什么是常量表达式：**

* **值不会改变**, (！！！生命周期内，值不会变)
* 且**编译期间**就能得到结果的表达式

```c++
const int max_files = 20; //max_files 是常量表达式，满足：值不会改变，且编译期间就能得到结果
const int limit = max_files + 1; // limit 是常量表达式
int staff_size = 27; // staff_size 并不是常量表达式，值会改变
const int sz = get_size(); // 不是常量表达式， 编译期间并不能得到结果。
```



**常量表达式的好处**

* 编译时候就能得到值，不用等到运行时候。
* 这样可以加快运行的 速度。



**constexpr提出来的目的就是：能在编译期搞定的事情，就不要拖到运行时。**



## constexpr 变量

C++11 新规定，可以将 变量声明为 **constexpr** 类型，以便由编译器来验证变量的是否为一个常量表达式。

* 声明为 **constexpr** 的变量一定 是一个常量
* **constexpr** 声明比 **const** 声明多的一个功能就是：验证这个 变量是不是 常量表达式。
* ？？？ 变量竟然称之为 常量表达式，真是不会起名字。。。。



```c++
constexpr int mf = 20; // 是个常量表达式
constexpr int limit = mf+1; // 是个常量表达式
constexpr int sz = size(); // 只有 size 是 constexpr 函数时，才是常量表达式
```



**可以定义为 constexpr 变量的类型有**

* 算术类型，引用，指针。（字面值类型）
* 自定义类需要**满足一定的条件**才能作为 字面值类型 。是不能作为 constexpr 的。



## constexpr 函数

> constexpr 函数是指 能用于常量表达式的 函数！！！！！！

定义 constexpr 需要遵循几条规定：

* 函数的 **形参** 与 **返回值的类型** 都必须是 **字面值类型** ！！
* 函数体内  **要么是空的，要么只能有** 一条 return 语句。（因为想着 编译期间就得到 返回值？？？？？）



关于 constexpr 函数：

* 允许函数 返回 **非常量表达式**



```c++
constexpr int exprdemo(int a, int b) {
    return a + b; // 函数体内只能有一条返回语句。
}

constexpr int exprdemo2(int a, int b) {
    int c = a+b;
    return c; // 只能包含 返回语句。。。
}
```

* 如果 实参都是 常量表达式，那么 返回常量表达式，就可以给 `constexpr` 变量赋值。
* 如果 实参不是 常量表达式，那么返回的就不是常量表达式，给 `constexpr` 赋值就会报错。



## constexpr 构造函数

**满足什么样条件的自定义类才能作为 字面值类型**

* 数据成员必须时字面值类型
* 类必须包含至少一个 `constexpr` 构造函数
* 如果一个 数据成员 函数有类内初始值， 则内置类型成员的初始值必须是一个常量表达式; (这个条件有点懵。。)
* 类必须 使用析构函数的默认定义。



**constexpr 构造函数**

* 函数体是空的
* 必须初始化所有的 数据成员
* 满足 `constexpr` 函数的基本条件
* 如果实参是 constexpr， 那么构建出来的对象就是 constexpr，如果 实参不是 constexpr ，那么构造出来的对象就不是 `constexpr`

```c++
class Debug {
public:
    constexpr Debug(bool hw, bool io) : hw(hw), io(io) {}
    constexpr Debug(bool flag) : hw(flag), io(flag) {}

private:
    bool hw;
    bool io;
};
int main(){
  constexpr Debug d{true, true}; // 是 constexpr 变量
  bool flag = true;
  constexpr Debug d2{flag}; // 不是 constexpr 变量，会报错。git
}
```





## 注意

* 内联函数 和 `constexpr` 函数都应该放在 头文件里。