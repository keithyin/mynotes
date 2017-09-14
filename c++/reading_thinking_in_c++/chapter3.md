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







## Tips

* 如果必须执行特定平台的活动，应当尽力将代码隔离在某一场所。C++中，经常把特定平台的活动封装在一个类中。
* ​



## Glossary 

*  函数原型（function prorotyping）：在声明和定义一个函数时，必须使用参数类型描述，这种描述就是“原型”。
* ​