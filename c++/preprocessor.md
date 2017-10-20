# C预处理器

预处理是在 程序编译之前进行的一步操作。



## 翻译程序

这个操作是 **预处理之前** 的操作，在 预处理 之前，编译器会对源代码会进行一些翻译操作：

1. 将源代码中出现的字符映射到 源字符集。
2. 查找 反斜线 (`\`) 后 紧跟  **换行符** (回车键产生的字符)的 实例，并删除这些实例。
3. 编译器将文本划分为：语言符号(`token`) 序列，空白字符序列，注释序列。（`token`：空格分隔的组）
   1. 编译器用一个 空白字符 代替 一个注释。
4. 然后，**程序进入预处理阶段**



```c++
cout << "hello \
world" << \
            endl;
// 根据 2, 上面的物理行 会被转换成
cout << "hello world" << endl;
```



```c++
int /*这里是注释*/ fox;
// 根据3， 上面的语句会转换成
int fox;
```



## 预处理器指令

* [`#define`](#define) 
* [可变参数宏](#可变参数宏)
* [`#include`](#include)
* [`#undef`](#undef)
* [`#ifdef`](#条件编译)
* [`#else`](#条件编译)
* [`#endif`](#条件编译)
* [`#ifndef`](#条件编译)
* [`#if`](#条件编译)
* [`#elif`](#条件编译)
* `#line`
* `#error`
* `#pragma`

预处理器指令 

* 都是由 `#` 开头
* ANSI 允许 `#` 与指令的其余部分有空格，但是**实际上并不行**。
* 由 `#` 开始，到第一个 换行符 为止，（指令的长度仅限于 **一行逻辑代码**）
* ​

## define

每个`#define` 行（逻辑行）由三部分组成：

* `#define` 自身
* 所选择的缩略语，这些 缩略语称为 宏（`macro`）

  * **宏的名字中不允许有空格，而且必须遵循C变量命名规则**
* 替换列表（replacement list）或叫 主体（body）， （这个地方可以省略，说明只是定义了这个一个宏）


预处理器在程序中发现了宏的实例后，总会用 主体 替换这个宏。

**宏展开：** 从宏变成最终文本的过程。



```c++
#define TWO 2
#define FOUR TWO*TWO

int main(){
  x = FOUR;
  return 0;
}
// 替换过程为
// x = TWO*TWO;
// x = 2*2;
```

* 从上面示例可以看出，宏定义中可以包含其他宏！
  * 一般而言，预处理器发现程序中的宏后，会用它的等价替代文本代替宏，如果该 文本中 还包括宏，则继续替换这些宏。
  * 如果**宏存在与双引号**内，则不予替换。

**语言符号**

从技术方面看，系统将 宏的 **主体** 当作语言符号(`token`)类型字符串，而不是字符型字符串。

C预处理器中的 语言符号 是宏定义主体中 **单独的词**（空格分割开的词）。

```c++
// 这个定义的主体中 只有一个语言符号（token）即 2*3
#define SIX 2*3

// 这个定义的主体中，有三个语言符号，2 * 4，主体中的空格看作 分割语言符号的  符号
#define EIGHT 2 * 4

// 这个和 上面那个是一样的，额外的空格不看做主体的一部分
#define EIGHT 2    *    4
```



**类函数宏**

```c++
// 括号一定要贴着 PRINT！！！！
#define POWER(x) x*x
```



**注意：**

* 宏的名字不能有空格，但是在 **替代字符串** 中可以有空格。
* **主体中**， 用圆括号 括住每个参数， 并括住整个主体。
* 用大写字母表示 宏的名字



## 可变参数宏

* 使用 `...` 和 `__VA_ARGS__`

```c++
#include <iostream>

using namespace std;
void add(int i, int j = 1) {
    cout << i + j << endl;
}

#define XNAME(...) add(__VA_ARGS__)

int main() {
    XNAME(1);
    XNAME(1, 4);
}
```



## include

预处理器发现 `#include` 指令后，就会寻找 文件名 并把 这个文件内容 包含到 当前文件中。被包含文件中的文本将替换源代码文件中的 `#include` 指令。

```c++ 
#include <iostream> // 尖括号代表 搜索系统目录
#include "myHeader.h" // 引号代表，先搜索当前目录，再搜索系统目录
#include "/usr/biff/p.h" // 搜索 /usr/biff 目录
```



## undef

用来取消 `#define` 的宏定义。

```c++
#define LIMIT 100
#undef LIMIT
```

宏的作用域 从 **#define** 开始，到 `#undef` 或文件尾 结束。



## 条件编译

`#ifdef, #else, #endif` 

```c++
#ifdef MAVIS //如果定义了，执行下面的语句，否则到 else 中执行
	#include "horces.h"
#else
	#include "birds.h"
#endif

#ifdef HH
 	#include "didi.h"
#endif //必须存在的
```



`#if, #elif`

```c++
# if SYS==1
	#define SYS_
#elif SYS==2
	#define SYS__
#endif
```

```c++
#define NE 10
int main() {
// 有无空格都可以
#if NE > 10
    cout << __DATE__ << endl;
}

#else
cout << __FILE__ << endl;
}

#endif
```






## C/C++ 宏中，`#` 与 `##` 的用法

### `#` 的作用

`#`的功能是将其后面的 **宏参数** 进行**字符串化操作**,  就是：宏变量替换后，左右各加一个双引号。（将语言符号转换成字符串！！！）

```c++
// #x 之间有无空格都可以
#define TEST(x) \
    #x
int main() {
    cout << "hello" << endl;
    cout << TEST("hello") << endl;
}

// 输出结果为：
// hello
// "hello"
// 可以看到 # 的作用是 加 双引号。
```



### `##` 的作用

`##` 称之为 连接符（`concatenator`），将两个语言符号组合成单个语言符号。这里连接的对象是 `Token` 就行，不一定非要是 宏变量。 

`N` 个 `##` 连接 `N+1` 个 `Token`。

```c++
// 注意，x##n 有无空格都可以
#define XNAME(n) x ## n
#include <iostream>
using namespace std;
int main() {
	// XNAME(4) 会变成 x4 
    int XNAME(4) = 3;
    cout << x4 <<endl;
}
```



## 预定义的宏

* `__VA_ARGS__` : 可变宏用到
* `__DATE__` ： 当前的时间
* `__FILE__`：当前文件名
* `__TIME__`： 源文件编译时间





## 参考资料

[C Primer Plus 5]  p446