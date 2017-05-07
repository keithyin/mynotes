# 引用和拷贝构造函数

`c++`中的指针：`int* -> void*`是不允许的，`void* -> int*`需要强制转换。

## c++中的引用
引用其实就看做一个变量的另一个名字。
```c++
int y = 12;
int &x = y;
//x就是y的另一个名字而已，访问x和访问y其实是访问相同的东西
```
下面列举几个注意事项：

* 当引用被创建时，必须被初始化
* 一个引用指向了一个对象后，就不能再更改
* 不能有NULL的引用

**常量引用：**
```c++
void g(const int &x){} //函数内不能修改x
```

```c++
void f(int **x){}
void f(int* &x){}//两个等价，指针的引用
```

## 拷贝构造函数
**按值传递的时候才会调用拷贝构造函数**

**当从现有的对象创建新对象时，拷贝构造函数就会被调用**

* 在函数传参的时候（值传递的时候）
* 函数返回的时候（值返回的时候）
* 赋值 （=）

```c++
class Name{
  string name;
public:
  //拷贝构造函数
  Name(const Name& name){
    name = name.name
  }
}
```

## 拷贝构造函数与 =

```c++
Mytype a; //调用构造函数
Mytype b = a; //同样是创建一个对象，调用拷贝构造函数
a = b; //并不是创建对象 operator = 被调用
```

**无论什么时候使用`=`来代替 普通形式的构造函数调用 来初始化一个对象时， 无论等号右侧是什么， 编译器都会找一个接受右侧类型的构造函数。但是应该避免，避免其它读者混淆**

## 临时对象
```c++
class Name{
  string name;
public:
  Name(string name):name(name){}
  string get_name(){
    return name;
  }
}

Name ge_name_obj(string name){
  Name tmp = Name(name)
  return tmp;
}

Name name = ge_name_obj("keith");//函数在返回的之前，会将tmp值复制到name对象中。
ge_name_obj("yin");
//那么对于这种情况呢，函数返回之前就会把tmp复制到临时对象之中，目
//的是可以满足型如下面的调用
ge_name_obj("world").get_name();
```
需要注意的：

* 临时对象的生存周期很短
* 临时对象是常量
