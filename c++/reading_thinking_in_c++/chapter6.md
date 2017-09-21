# chapter 6

**构造函数与析构函数**



**C++ 为自定义的数据类型提供了类型检查保证**



## 构造函数保证对象正确的初始化

**形式** : 和类名一样

```c++
class X{
  public:
  X(){}
  X(int i){
    val = i;
  }
  private:
  int val;
}

void main(){
  X a; // 这么定义对象，编译器会自动调用 构造函数
  X b(4); // 这么调用带参数的 构造函数
  X* c = new X; // new X(); 等价
  X* d = new X(10);
}

```



## 析构函数确保资源回收

> 确保资源的回收，内存 等等。。。

**形式**: `~类名`

```c++
class X{
  public:
  X(){}
  X(int i){
    val = i;
  }
  ~X(){
    // 执行一些回收操作， 析构函数不带任何参数
  }
  private:
  int val;
}
```



**什么时候会调用 析构函数**

* 当对象超出它的作用域时
* delete 的时候也会调用？



## 聚合初始化

语法 ： `{}`

聚合( `aggregate` )类型:

* 数组
* struct
* class

```c++
// 数组
int a[5] = {1, 2, 3, 4, 5};

// struct
struct Demo{
  int a;
  float b;
  char c;
}

Demo demo = {1, 1., 'c'};

// class
class X {
public:
    X() {}

    X(int a, int b) {
        val = a + b;
    }

private:
    int val;
};

X x = {1, 2}; // 会找相应的 构造函数

// struct 数组和 class 数组 也可以用，就是 {} 嵌套

```

