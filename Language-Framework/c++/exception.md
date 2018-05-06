# C++ 异常处理



`C++` 异常处理建立在三个关键字之上：

*  `throw` : 抛出异常
* `try` ： 
* `catch`： 捕获异常

```c++
#include <iostream>

double division (int a, int b) {
  if (b==0) throw "divide 0 error";
  return static_cast<double> (a) / b;
}


int main() {
  using namespace std;
  int a = 9;
  int b = 0;
  double res;
  try {
    res = division(a, b);
    cout << res << endl;
  } catch (const char *info) {
    // 抛出 const char * 类型的异常的时候会进入这个处理块。
    cout << info << endl;
  }
  // 无论有无异常，都会执行这部分。
  cout << "hello world" << endl;
  return 0;
}
```



**如果异常没有及时的进行处理，它会一层的一层的向上传递，如果在 main 中还没有处理, 程序会在抛出异常的地方终止。**

```c++
#include <iostream>

double division (int a, int b) {
  if (b==0) throw "divide 0 error";
  return static_cast<double> (a) / b;
}

double call() {
  int a = 1;
  int b = 0;
  double result;
  result = division(a, b);
  return result;
}

int main() {
  using namespace std;
  try {
    double res = call();
  } catch (const char* info) {
    cout << info << endl;
  }
  cout << "hello" << endl;
  return 0;
}
```



**`C++` 也提供了很多标准异常，在 `<exception>` 头文件中：**

![](imgs/cpp_exceptions.jpg)

**如何自定义 异常**

* 继承 `exception`
* 重写 `what()`

```c++
#include <iostream>
#include <exception>
using namespace std;

struct MyException : public exception {
  // what() 是 exception 提供的虚函数， 子类应该重写之，用来打印错误信息。
   const char * what () const noexcept {
      return "C++ Exception";
   }
};
 
int main() {
   try {
      throw MyException();
   } catch(MyException& e) {
      std::cout << "MyException caught" << std::endl;
      std::cout << e.what() << std::endl;
   } catch(std::exception& e) {
      //Other errors
   }
}
```











## 参考资料

[https://www.tutorialspoint.com/cplusplus/cpp_exceptions_handling.htm](https://www.tutorialspoint.com/cplusplus/cpp_exceptions_handling.htm)

[https://www.geeksforgeeks.org/exception-handling-c/](https://www.geeksforgeeks.org/exception-handling-c/)

[https://stackoverflow.com/questions/12833241/difference-between-c03-throw-specifier-c11-noexcept](https://stackoverflow.com/questions/12833241/difference-between-c03-throw-specifier-c11-noexcept)

