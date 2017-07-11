# c++ class 基础

**需要注意的几点**

* 类中定义的方法，调用不用方法的先后顺序。

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
