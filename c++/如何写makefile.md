# 如何写 makefile

makefile 是一个组织代码编译的工具



## 一个简单的例子

假设现在我们有三个文件，`hellomake.c`（主函数文件）， `hellofunc.c`（函数文件）， `hellomake.h`（头文件）。

```c++
// hellomake.c 主函数文件
#include <hellomake.h>

int main() {
  // call a function in another file
  myPrintHelloMake();

  return(0);
}
```



```c++
// hellofunc.c函数文件
#include <stdio.h>
#include <hellomake.h>

void myPrintHelloMake(void) {

  printf("Hello makefiles!\n");

  return;
}
```



```c++
// hellomake.h  头文件 
/*
example include file
*/

void myPrintHelloMake(void);
```



通常情况下： 我们会使用以下指令来编译代码：

```shell
gcc -o hellomake hellomake.c hellofunc.c -I.
```

* 将两个 `.c` 文件编译成 一个可执行文件 `hellomake`。
* `-I. `: 告诉 `gcc` 在找 头文件的时候要看一下当前文件夹。 
* 在不用 makefile 的情况下，在 `test/modify/debug` 循环中，我们要不停的执行这条 `gcc 指令`，但是一旦找不到这条 `gcc 指令`了，那就只能重新敲一遍了，（哭死。。）所以，makefile 还是必须要用的。



我们可以为上面的代码写一个 `makefile`：

```shell
hellomake: hellomake.c hellofunc.c
     gcc -o hellomake hellomake.c hellofunc.c -I.
```

* 文件名字可以是 `Makefile` 或 `makefile`。
* 当写好这个文件后，然后在命令行 键入 `make` ，它就会执行在 `makefile`中定义好的编译命令。




**makefile文件的简单格式：**

```shell
<target>: [ <dependency > ]*
   [ <TAB> <command> <endl> ]+
```

* target : 通常是一个文件
* dependence:  target 依赖的文件
* 要运行的命令，基于 target 和  依赖




## 参考资料

[http://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/](http://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/)

[https://www.cs.umd.edu/class/fall2002/cmsc214/Tutorial/makefile.html](https://www.cs.umd.edu/class/fall2002/cmsc214/Tutorial/makefile.html) 