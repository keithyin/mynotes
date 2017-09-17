# chapter3



**函数与。。。**



## 文章阅读

### 函数

```c++
void func(int, int, int); //这种形式在声明是可以的
void func2(int a, int b){
  // 在定义的时候，就需要加上参数的名字，不然在函数体内没法使用参数的
}
// 注意：在 C++ 中，定义的时候 参数列表中 可以包含未命名的参数，这个参数在函数体内是无法使用的，占坑而已

// 下面这俩都代表 函数需要 0 个参数
void func3();
void func4(void);
```



**return 语句退出函数返回到函数调用后的那一点。**



×××××××××××××××××××××××××××××××××××

关于函数欠缺的部分有：

* 变参列表
* 函数指针
* std::function
* ...........





## 控制语句

* if--else
* while
* do-while
* for
* switch
* goto （不要用）

```c++
// 代码块中只有一句，不需要加大括号，如果有多句，就需要加大括号了
if (expression)
  // do something;
else
  // do something;

if (experssion)
  // do something;
else if (expression)
  // do something;
else
  // do something;

// 循环语句, 每次在进入 代码块之前计算 expression, true 则进入，false则跳过
while (expression){
  // do something;
}

do{
  // do something;
}while (expression); //先执行，然后判断下次要不要继续执行

for(initialization; conditional; step){
  // do something;
  // 第一次执行循环是：初始化，判断 conditional 是否满足，循环末尾执行 step。
  // 然后是：判断 conditional 是否满足，循环末尾执行 step。
  // 表达式中的 initialization，conditional，step 都可以为空
}


// selector 是一个产生 整数值的选择器
switch(selector){
  case integral_value1: statement; break;
  case integral_value2: statement; break;
  case integral_value3: statement; break;
  ...
  default: statement; break; 
}
```



**break 与 continue**

* break : 从循环体中跳出来
* continue: 结束本次循环，进行下次循环






## 递归

> 函数调函数

```c++
void func(){
  // do something;
  func();
  // do something;
}
```



## 自增与自减

```c++
a++; // 先返回 再 加
--a; // 先加 再返回
```



## 指针

不管什么时候运行一个程序，都是首先把它装入（一般从磁盘装入）计算机内存。因此，程序中的所有元素都驻留在内存的某处。

**由于程序运行时驻留在内存中，所以程序中的每一个元素都有地址**

* 取地址运算符 ： `&` （作为右值时，是取地址运算符，作为左值，是引用运算符。）
* C/C++ 有一个专门的 存放地址的变量类型，指针！



```c++
int *ipa, ipb, ipc; // 只有第一个是指针
```



## 函数传值

**向函数传递参数时，在函数内部生成该参数的一个拷贝**

* 值传递
* 指针传递（也相当于 值传递）
* 引用传递



## 作用域

变量的作用域由 变量所在的 **最近一对括号** 确定。



**全局变量：** 在所有函数体外部定义的。



## 常量

* 预处理器
* const 

```c++
#define PI 3.1415926
const int x = 10;
```



## 逗号运算符

* 定义多个变量时用来分隔变量
* 用于分隔表达式，只产生最后一个表达式的值(只把最后一个表达式的值作为右值)

```c++
int a=0, b=0, c=0;
a = (b++, c++);
```



## 类型转换

```c++
int a = (int) b; // 第一种语法

```








## Tips

* 如果必须执行特定平台的活动，应当尽力将代码隔离在某一场所。C++中，经常把特定平台的活动封装在一个类中。
* ​




## Key Word

* typedef : 类型名称重定义




## Glossary 

*  函数原型（function prorotyping）：在声明和定义一个函数时，必须使用参数类型描述，这种描述就是“原型”。
*  ​