# C++初探



**namespace**

* `namespace` 是管理变量名的，一种简单啊的理解可以是，命名空间中的 变量与函数 的名字都加了个前缀

```c++
namespace hello{
  void say_hello(){
    cout<<"hello"<<endl;
  }
}

using namespace std; //这句话会导致编译错误，因为会将 hello::say_hello 暴露出来

void say_hello(){
  cout << "hello out" << endl;
}
```



**enum**

* 可以在全局空间中定义，可以在函数定义，可以在类中定义。

```c++
int main(){
  enum AGES {one, two, three};
  AGES ages = one;
  // ...
}
```



