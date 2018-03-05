# 谷歌C++编码规范总结



## 命名约定

```c++
/* 文件名命名
my_useful_class.cc
my-useful-class.cc
myusefulclass.cc 
以上三个都可以
*/

// 普通变量命名，不要大小写混合
int price_count_reader; 
int *price;

// 类数据成员， 屁股上要带个 _
class Demo {
private:
  string table_name_;
  int *data_;
};

// 结构体变量, 和普通变量命名一样
struct UrlTableProperties {
  string name;
  int num_entries;
  static Pool<UrlTableProperties>* pool;
};

// 类名, 首字母大写，驼峰命名法
class DemoClass {};

/* 函数命名
普通函数，类成员函数： 首字母大写，驼峰命名法
set 和 get ： 与变量名要一致
*/
void Count(); // 普通函数 或 类成员函数
void set_count(); // set 函数。
int count(); // 取值函数 和变量名一致即可。


// 命名空间，使用小写字母 加下划线命名
namespace index {}
namespace index_util {}

// 常量命名，用 k 做前缀
const int kDaysInWeek;

// 枚举命名
enum UrlTableErrors {
    kOK = 0,
    kErrorOutOfMemory,
    kErrorMalformedInput,
};
enum AlternateUrlTableErrors {
    OK = 0,
    OUT_OF_MEMORY = 1,
    MALFORMED_INPUT = 2,
};


```

