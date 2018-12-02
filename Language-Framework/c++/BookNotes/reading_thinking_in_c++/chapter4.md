# chapter 4



## 动态存储分配

**C : **

* malloc
* free



**C++ : **

* new : 语法： `new Type` ,返回的是指向分配的空间的指针
* delete : 语法 `delete pointer` ， pointer 是 new 时 返回的指针

```c++
int * a = new int;
delete a; // 传给地址就好

int *b = new int[10]; //分配的是数组，b 是指向数组第一个元素的地址
delete []b; //delete 时记得加个 []， 让编译器知道 b 是个指向数组的指针，而不是指向值的指针
```



**C++允许将任何类型的指针赋给 `void*`, 但不允许将 `void*` 指针赋给其它类型的指针**



## 类型

```c++
s.add(i); // 向 s 发送消息。对一个对象 调用一个成员函数
```



## 全局作用域解析

**`::` 作用域运算符**

```c++
::a ; // :: 代表的是全局作用域

// 在类的的 成员函数定义的时候也会看到这个 运算符, 用来指定是哪个作用域下的 name() 
void Person::name(){
  
}
```





## 疑问

* 当创建一个对象时，对象是怎么找到 成员函数的？







## 一些规定

* C/C++ 中都可以重声明函数，但是不可以重声明 `struct, class`
* `#endif ` 后面无注释是不符合规定的。 `#endif // HEADER_FLAG`
* 在头文件中最好不要使用 `using` 指令
* ​



## Glossary

* 封装（encapsulation）： 将**数据**连同**函数**捆绑在一起。
* ​