# 链接指示，extern "C"

* C++ 程序 **调用其它语言** 编写的函数，（动态链接/静态链接）
  * 这时C++ 需要在头文件中 **声明** 函数原型
  * 这是因为，不同的编译器，函数名编译之后的结果不一样
  * 导入其它语言函数的话，只需要 声名的时候加上 `extern "C"` 即可

```c++
/*
extern "C" / "Ada" / "FORTRAN"
	不能出现在 类定义或函数定义内部
	
	链接指示必须在函数的每个声明中都出现

*/


// 在C++中 声明一个非 C++ 的函数
extern "C" size_t strlen(const char*); //单语句

extern "C" {                           // 复合语句
  int strcmp(const char*, const char*);
  char *strcat(char*, const char*);
}

// 多重声明的形式可以用于整个头文件
extern "C"{
  #include <string.h>
}

// 指向 extern "C" 函数的指针，前面价格 extern "C" 就可以了
extern "C" void (*pf) (int);

// 链接指示 对 函数，（形参，返回值）的函数指针 都有效
```



* 导出 C++ 函数到其它语言
  * 使用链接指示对函数进行**定义**，可以令一个 C++ 函数在其它语言编写的程序中可用
  * 编译器将为 指定的函数 生成适用于指定语言的代码
  * 导出的话，声明定义都需要 `extern "C"`

```c++
extern "C" double calc(double dparm);
extern "C" double calc(double dparm){
  
}
```

