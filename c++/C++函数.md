# C++函数



* 关于C++函数默认参数需要注意的

  * 需要在声明是指定默认参数，且在定义时不用说明

  ```c++
  void add(int i, int j=1);

  void add(int i, int j){ //如果在定义的时候也写上int j=1的话，会报错。
    return i+j;
  }
  ```

* C++不支持关键字参数调用

  * 用了python之后老是感觉c++也可以关键字调用，但是实际上是不可以的

  ```c++
  add(i = 1, j = 2); //这样是错误的
  ```

## 函数指针



```c++
// 定义：func 是一个函数指针，指向一个 参数列表为空，返回值为void 的函数
void (*func)();


// printf 是一个函数指针，等价与 &printf()
void printf(){
  a++;
}

int main(){
  void (*fp)();
  fp = printf;
  (*fp)(); // 调用。
}
```

