# 标准模板库

* 容器：放任何类型的数据
* 迭代器：迭代器就像是指针



## vector

* 调用对象的 **拷贝构造函数**





## string

* `string` 与 `char*`
  * `string` 封装了字符指针

```c++
#include <string>
using namespace std;
int main(){
    string a = "hello"; // char* --> string
    string b("bbbb");
    b.c_str(); // 返回内存首地址 string --> char*
    
    // 字符串的连接
    b = b+a;
    b.append(a);
    
    // 查找和替换
    
}
```

