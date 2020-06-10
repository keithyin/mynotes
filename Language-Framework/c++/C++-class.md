# c++ class 基础

**需要注意的几点**

* 类中定义的方法，调用不用考虑方法的先后顺序。
* 这是因为 C++ 标准。

```c++
class LinkStack{
public:
	struct Node{
		int data;
		Node* next;
	}; // 这里建一个结构体。
	LinkStack(){
		top_node = new Node;
		top_node->data = NULL;
		top_node->next = nullptr;
	}

	bool push(int val){
		Node* new_node = new Node;
		new_node->data = val;
		new_node->next = top_node;
		top_node = new_node;
		return true;
	}
	bool pop(int &val){
		if (is_empty()){ // 虽然在这后面才定义，但是在这就可以调用，为啥？？
			val = 0;
			cout << "the stack is empty" << endl;
			return false;
		}
		val = top_node->data;
		Node* old_node = top_node;
		top_node = top_node->next;
		delete old_node;
	}
	bool is_empty(){
		if (top_node->next == nullptr)
			return true;
		return false;
	}

private:
	Node* top_node; // point to the top node, not the top+1
};

int main(){
  LinkStack::Node node; //类中类可以这么用。
}
```

**delete**

```c++
Stack* stack = new Stack;
cout << stack << endl;
// 删除的仅是 stack 指向的对象
delete stack;
// stack 保存的指针还是有的
cout << stack << endl;
```





```c++
class Age {
public:
    Base *b = new Base;
    int i = 0; // 这里可以给初始值
    const int age = 10; //这里可以给初始值

    Age() {
        cout << "Age Constructor" << endl;
    }
};
```







## 语法

```c++
class Name{
public:
  Name();
  ~Name();
  string get_name();
  void print_name();
private:
  string name;
}
```

基本构成部分：

- 构造函数

- 析构函数

- 拷贝构造函数

- 其他函数与属性

## class中常量处理

- 在类定义时，只能给static const 成员属性赋初值，其它都不行。

- 对于 const：需要在构造函数初始化列表中初始化

- static const：必须在定义的时候初始化，可以看作为编译期间常量

- 可以使用无标记 enum 代替 static const。
```c++
class Bunch{
  enum{size=1000};
  int i[size];
}
```

## const 对象与成员函数

什么是`const`对象：
```c++
const Name name;//const对象，对象的数据成员在生命期内不会被改变
```
**只能调用`const`对象的const成员函数！！**

为什么呢？

因为编译器通过成员函数是否为`const`来得知此成员函数是否更改了对象的数据成员。

```c++
class X{
  int i; //class 默认为private
public:
  X(int ii);
  int f() const; //注意和返回值为常量的函数的区别。
}

X::X(int ii):i(ii){}
int X::f() const {return i}
```

## 内联函数
内联函数：能够像普通函数一样具有我们所有期望的任何行为，与普通函数不同之处在于，**内联函数在适当的地方像宏一样展开，所以不需要调用函数的开销。**

```c++
//函数声明和定义要写在一起！！
inline int plusOne(int x){
  return ++x;
}
//一定要把内联函数放在头文件里

//类内部定义的函数自动成为内联函数
class Name{
  string name;
public:
  Name(string nname):name(nname){}//内联
  ~Name(){}//内联
  void print(){cout<<name<<endl;}//内联
}
```

## c++类中，如果你不干编译器就帮你干的几件事
**如果你不干编译器就帮你干的，如果你干了，编译器就不帮你干的几件事**

* 构造函数
* 析构函数
* 拷贝构造函数
* type::operator=(type)



## 对象创建

```c++
People people("name", 22);
People people = People("name", 22);
People *people = new People("name", 22);
```





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

**无论什么时候使用`=`来代替 普通形式的构造函数调用 来初始化一个对象时， 无论等号右侧是什么， 编译器都会找一个接受右侧类型的构造函数。但是应该避免，避免其它读者混淆。**

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





# C++ class 总结



## 访问属性

* **public :**  类内 可访问， 类的用户可访问
* **protected :** 类内可访问， 类的用户不可访问
  * 派生类类内可访问
  * 派生类的 友元 可以访问 派生类的 protected 属性，不可访问 基类的 protected 属性。 
* **private :** 类内 可访问，类的 用户不可访问。
  * 派生类 类内不可访问， 派生类的用户不可访问。



## 基本语法





## 常量对象？？？



## 继承

**语法：**

```c++
class Y : public X{
  
};// 这是一个声明语句，所以后面要加 分号， 语句块的话就不需要加分号。
```

**Y继承 X：**

* Y 将 包含 X 中所有 数据成员和成员函数。正如没有继承 X， 直接在 Y 中创建一个 X 的成员一样，Y 是包含了一个 X 的子对象。
* 继承 依旧 遵守保护机制。即：`X` 中的私有成员 在 `Y` 中不能直接访问。



**几种继承**

* **public 继承** : 继承下来的 数据成员和 成员函数 保持之前的访问属性。
* **protected 继承**：继承下来的 public 变成了 protected
* **private 继承：** 继承下来的 数据成员 和 成员函数 的访问属性都变成 私有，即 `Y` 内可访问，`Y` 外不可直接访问。



**重写：override**

* 重新 定义 基类继承下来的 成员。



## 多态

**多态： 即， 晚捆绑， 通过 虚函数 实现**

**为了实现 晚捆绑，C++ 要求在基类 `声明这个函数时` 使用 virtual 关键字**

* 晚捆绑仅对  `virtual` 函数起作用
* 只在使用 含有 `virtual` 函数的基类的地址时发生。
* 仅仅需要在声明是 使用 `virtual`， 定义时并不需要。仅仅需要在 基类中声明 `virtual` 函数。 
* 派生类中 `virtual` 函数的重定义 常常称为 重写 (`override`)



**C++如何实现 晚捆绑**

* 对每个 包含虚函数的 类创建一个 表 (`VTABLE`)
* 在 `VTABLE` 中，编译器放置 特定类的 虚函数地址。
* 在每个 带有虚函数的 类中， 编译器秘密地放置一个指针，称为 `vpointer`， 指向这个对象的 `VTABLE`
* 当通过 **基类指针做虚函数调用时** ，编译器静态插入地 插入 **能取得 这个 `VPTR` 并在 `VTABLE` 表中查找函数地址的** 代码. 这样就能正确的调用函数 并引起晚捆绑的发生。





```c++
#include <iostream>

using namespace std;

class X {
public:
    virtual void print() {
        cout << "i am in X" << endl;
    }

    void call_print() {
        // 等价于 this->print();
        print();
    }
};

class Y : public X {
public:
    void print() override {
        cout << "i am in Y" << endl;
    }
};

int main() {
    X *x = new Y;
    x->call_print();

    std::cout << "Hello, World!" << std::endl;
    return 0;
} // 输出： i am in Y
```



## 删除编译器提供的默认 方法

以下方法, 如果用户没有显式实现, 编译器则会提供一个对应的默认方法

* 构造函数: `A(){}`
* 拷贝构造函数`A(const A &a)`
* 赋值函数 `A& operator=(const A &rhs)`
* 析构函数 `~A()`

如何我们并不想编译器默认生成这些方法的话, 可以使用 `=delete` 关键字

* `A()=delete;`



## 如何避免隐式类型转换

C++ 会有一些隐式类型转换的场景, 比如:

* 赋值的时候
* 传参的时候

如果不想用隐式类型转换的话, 可以给构造函数加一个 `explicit` 关键字



