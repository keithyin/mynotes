* 除非对象是右值的, 否则所有的 stl 容器存储的都是对象的副本
* 当从容器中获取一个对象的时候, 得到的将是一个容器对象的引用.
* 迭代器所指向的对象必须是可交换的.



# 非关联容器

* [array](http://en.cppreference.com/w/cpp/container/array) 静态连续 数组
* [vector](http://en.cppreference.com/w/cpp/container/vector) 动态连续数组
* [deque](http://en.cppreference.com/w/cpp/container/deque) `double-ended queue` 双端队列
* [forward_list](http://en.cppreference.com/w/cpp/container/forward_list) 单向链表
* [list](http://en.cppreference.com/w/cpp/container/list) 双向链表vector

### array

```c++
#include <array>
```

* **固定大小**的序列容器



### vector

```c++
#include <vector>
```

**特点：**

- **数据连续存放**
  - 具体实现是, 如果超过了当前的容量, 会重新进行内存分配,  然后将之前的值搞过去
- 可以动态调整大小
  - 2倍速增大, `1/4` 速度减小 (`似乎是这样, 不大确定`)
- 不存在包含引用的 `vector`, **不能存储引用** , 但是可以存储对应对象的 `指针` 或者 `shared_ptr`

**API 简介：**

- `push_back`: 
  - 对于临时对象，调用对象的移动构造函数
  - 对于非临时对象，调用对象的复制构造函数
- `pop_back()` ：会析构对象
- `back()` : 返回的是引用。
- `=` : 
  - vector 的 `=` 中，调用的是里面所包含对象的复制构造函数
  - vector 的移动赋值操作中，直接移动 `vector` 内部的资源，不会调用所包含对象的任何函数



标准库类型`vector`表示对象的集合，其中所有对象的类型都相同，集合中的每个对象都有一个与之对应的索引，索引用于访问对象。

想使用`vector`的话，首先要包含头文件`<vector>`。



```c++
// 创建对象
vector<T> v1; //v1是一个空vector，它潜在的元素是T类型的，执行默认初始化
vector<T> v2(v1); // v2中包含v1中所有元素的副本
vector<T> v2 = v1; //等价于v2(v1)
vector<T> v3(n,val); //v3包含了n个重复的元素，每个元素的值都为val
vector<T> v4(n); //v4包含了n个重复地执行了值初始化的对象
vector<T> v5{a,b,v,d,...}; //v5包含了初始值个数的元素，每个元素都被赋予了相应的初始值
vector<T> v5 = {a,b,c,d,...}; // 等价于v5{a,b,v,d,...}

// public function
back(); // 返回最后一个值
pop_back(); // 弹出最后一个值
v.empty(); //如果v中不含有任何元素，返回真，否则，返回假
v.size(); //返回v中元素的个数
v.push_back(t); //想c的尾端添加一个值为t的元素
v[n]; //返回v中第n个位置元素的引用
v1=v2; // 用v2中的元素拷贝替换v1中的元素
v1={a,b,c,d,...}; //用列表中的元素拷贝替换v1中的值
v1 == v2; //v1和v2相等，当且仅当它们的元素数量相等且对应位置的元素值都相同
v1 != v2; //
<, <=, >, >=; // 以字典顺序进行比较


// 高级遍历
for (auto &a : vec) {
  cout << a << endl;
  a = new_val; // 可以改值。
}
```

```c++
std::vector<std::string> words;

// 这里是创建了一个临时对象, 然后会调用 移动版的 push_back 函数
words.push_back(std::string("hello"));

// 这时 编译器会 生成一个临时的 string 对象, 然后 会调用移动版的 函数
words.push_back("hello");

// 直接 inplace 执行构造函数, emplace_back 传的参数是 构造函数所需要的参数
words.emplace_back("hello");
```





### deque

* 双向 `queue`: 可以动态增加, 缩减
* 没有保证存储在连续的空间中

### list

* 双向链表

### forward_list

* 单向链表

|                      | 存储       | 优点                         | 缺点                     |
| -------------------- | ---------- | ---------------------------- | ------------------------ |
| array                | 连续空间   |                              |                          |
| vector               | 连续空间   | 下标索引快;在后面添加快      | 在前面添加慢             |
| deque                | 非连续空间 | 可以下标索引;前后插入都快;   | 不能通过位置偏移进行索引 |
| list                 |            | 任何位置的插入和删除都快     | 仅支持双向顺序访问       |
| forward_list         |            | 热和位置的插入和删除都快     | 仅支持单向顺序访问       |
| string(与vector类似) |            | 可以下标索引; 尾部插入删除快 |                          |

* **Note** : 下标索引 等价于 随机访问



# 关联容器

> 关联容器支持高效的关键字查找与访问。

**有序** (`key` 必须定义 `<`, 这样才能比较) , 这里应该是用的红黑树存储的

> 注意:这个有序, 并不是插入的顺序, 而是大小顺序, 因为是用红黑树存储的, 所以遍历是大小的顺序出来的.

- [set](http://en.cppreference.com/w/cpp/container/set) 
- [map](http://en.cppreference.com/w/cpp/container/map)
- [multiset](http://en.cppreference.com/w/cpp/container/multiset)
- [multimap](http://en.cppreference.com/w/cpp/container/multimap)

**无序** (使用 hash 方式存储), 

>  关键字满足两个条件: 可 `hash`; 有 `==` 运算

- [unordered_set](http://en.cppreference.com/w/cpp/container/unordered_set)
- [unordered_map](http://en.cppreference.com/w/cpp/container/unordered_map)
- [unordered_multiset](http://en.cppreference.com/w/cpp/container/unordered_multiset)
- [unordered_multimap](http://en.cppreference.com/w/cpp/container/unordered_multimap)



**特点**

* `map` : `key-value`，保存的是键值对，通过 `key` 来找 `value` 是非常快的。
  * `key` 需要定义 `<` 运算符。
  * 存放的数据是通过 `key` 排序好的
  * 比 `unordered_map` 速度要慢，因为要排序
* `set` : 保存的仅仅是键 `key`，找一个 `key` 是否在 `set` 中是相当快的。

```c++
#include <map>
map, multimap;

#include <set>
set, multiset;

#include <unordered_map>
unordered_map, unordered_multimap;

#include <unordered_set>
unordered_set, unordered_multiset;

#include <utility>
pair;
```



使用 `map`

```c++
#include <map>
#include <string>
#include <iostream>
using namespace std;

int main(){
  map<string, size_t> word_count;
  string word;
  while(cin>>word){
    ++word_count[word];
  }
  
  for (const auto &w: word_count) // w 是 pair 类型
    cout<<w.first<< " occurs " <<w.second<< " times."<<endl;
}
```



**关联容器：关键字类型的要求**

* 必须定义关键字的比较方法 (`定义好 < 就可以了`)

### pair

> map 中保存的值是 pair。

```c++
#include <utility>
using namespace std;

int main()
{
  pair<T1, T2> p;
  pair<T1, T2> p(v1, v2);
  pair<T1, T2> p = {v1, v2};
  make_pair(v1, v2); //返回一个用 v1,v2 初始化的 pair，pair 的类型会推断出来。
  
  p.first; // pair 的第一个数据成员
  p.second; // pair 的第二个数据成员
  
  p1 == p2; // 
  p1 != p2; // 
}
```



###关联容器的操作

**关联容器额外的类型别名**

* `key_type`: 关键字类型
* `mapped_type`: 映射的类型，只在 map 中可用
* `value_type`: 容器中保存的值的类型，map中是 pair



**迭代器：**

> map 中的 pair，第一个关键字是 常量。
>
> 迭代器是指向容器中元素的*指针*。
>
> map 中的元素排放顺序，是按照 key 的大小 从小到大排序的。

```c++
auto map_it = word_count.begin(); //返回的是一个指针，指向容器中的第一个元素。
// map<string,int>::iterator map_it = word_count.begin();

cout<< map_it->first<<endl;
cout << map_it->second<<endl;

map_it->first = "new key"; //错误，因为key为常量
```

> set 中的 key 也为常量



**添加元素：**

```c++
// 向 map 中添加元素
word_count.insert({"new_word",1});
word_count.insert(make_pair("new_word",1));
word_count.insert(pair<string, int>("new_word", 1));
word_count.insert(map<string, int>::value_type("new_word", 1));

// 除了 insert 外，还有 emplace
// insert(b,e) b,e为迭代器。哪里开始，哪里结束。
```



**删除元素：**

```c++
c.erase(k); // 删除指定key 的元素
c.erase(p); // 删除迭代器 p 指向的元素
c.erase(b,e); // b，e 为迭代器，开始与结束。
// 返回 0,1... 表示删除了几个
```



**查找元素**

```c++
c.find(key); //会返回一个指针, 如果返回的不是end, 就是找到咯
```





**map的下标操作：**

> map 和 unordered_map 提供了下标运算符和一个对应的 at 函数。set 不支持下标运算符。

```c++
map<string, int> word_count;

word_count["Anna"] = 1;

c[k]; //如果不存在，则插入。
c.at(k); //如果不存在，会跑出 out of range 异常

// c[k] 与 at(k) 返回的都是 value 的引用，这样，才可以在外面修改他们。
```

下标运算符将会执行以下操作：

1. 在 word_count 中搜索关键字为 `Anna` 的元素，找到则赋值，没找到的话
2. 将一个新的 key-value 插入到 word_count 中。关键字是 const string `Anna` ，值进行值初始化。
3. 提取出新插入的元素，并将值 1 赋给它



**访问元素**

```c++
c.find(key); // 存在，返回指向元素的迭代器。不存在，返回指向尾部的迭代器。
c.count(key); // 统计关键字为 key 的数量

c.lower_bound(k); //返回一个迭代器，指向第一个关键字不小于 k 的元素
c.upper_bound(k); //指向第一个大于 k 的元素
e.equal_range(k); // 返回一个迭代器 pair，表示关键字 等于 k 的元素的范围。
```



# Container adaptors

- [stack](http://en.cppreference.com/w/cpp/container/stack)
- [queue](http://en.cppreference.com/w/cpp/container/queue)
- [priority_queue](http://en.cppreference.com/w/cpp/container/priority_queue)

### priority_queue

* 高优在前, 优先级如何计算需要我们定义

```c++
std::priority_queue<std::string> words;//声明一个空的

std::string words[] = {"one", "two", "three"};
std::priority_queue<std::string> words {std::begin(words), std::end(words)};

// 第一个是 数据类型, 第二个是用什么 方式存储, 第三个是 优先级的判别方法
std::priority_queue<std::string, std::vector<std::string>, std::greater<std::string>> words = words {std::begin(words), std::end(words)};


```

* `std::greater` 会使用对象的 `operator>()` 进行比较
* `std::less` 会是用 对象的 `operator<()` 进行比较
* 内部计算, 元素比较, `compare(a, b)`, 如果返回 `true` , a 就往后站



# heap

* 不是容器, 而是一种特别的数据组织方式
* `priority_queue` 其实是堆的一个简单封装
* 创建堆

```c++
//创建堆
std::vector<double> numbers{1.0, 4.2, 6.3, 0.1};
// 对随机访问迭代器指定的一段元素进行重新排列, 默认是用 < , 生成一个大顶堆
std::make_heap(std::begin(numbers), std::end(numbers));

// 普通数组也可以make_heap, 通过传入第三个参数, 使用 > 号, 所以生成的是一个小顶堆
std::make_heap(std::begin(numbers), std::end(numbers), std::greater<>());
```

* 插入元素
  * 先往原始容器中使用 **`push_back`** 添加元素!!!!
  * 然后再调用`push_heap` 重排 (一定是`push_back`添加元素哦, 因为 `push_heap` 会认为最后一个是新加的.)

```c++
std::vector<double> numbers{1.0, 4.2, 6.3, 0.1};
std::make_heap(std::begin(numbers), std::end(numbers));
numbers.push_back(19.0);
std::push_heap(std::begin(numbers), std::end(numbers))
```

* 移除元素
  * 先调用`pop_heap` , 会将第一个元素移动到最后, 并且保证剩下的依旧是一个堆
  * 然后可以使用容器的 `pop_back` 方法来 移除

```c++
std::vector<double> numbers{1.0, 4.2, 6.3, 0.1};
std::make_heap(std::begin(numbers), std::end(numbers));
std::pop_heap(std::begin(numbers), std::end(numbers));
numers.pop_back();
```

* 工具函数, 确定是不是堆: `std::is_heap`



# 如果容器中保存的是指针, 在使用 priority_queue的时候如何指定比较函数呢

```c++
auto comp = [](const shared_ptr<string> &wp1, const shared_ptr<string> &wp2){return *wp1 < *wp2;};

// 第三个模板参数 传入 比较的参数签名.
std::priority_queue<shapred_ptr<string>, std::vector<shared_ptr<string>, decltype(comp)> words(comp, some_vector);

```





#参考资料

[http://en.cppreference.com/w/cpp/container](http://en.cppreference.com/w/cpp/container)