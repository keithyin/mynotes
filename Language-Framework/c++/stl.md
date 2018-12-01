# c++ 标准模板库



## 序列容器

* [array](http://en.cppreference.com/w/cpp/container/array) 静态连续 数组
* [vector](http://en.cppreference.com/w/cpp/container/vector) 动态连续数组
* [deque](http://en.cppreference.com/w/cpp/container/deque) `double-ended queue` 双端队列
* [forward_list](http://en.cppreference.com/w/cpp/container/forward_list) 单向链表
* [list](http://en.cppreference.com/w/cpp/container/list) 双向链表



## 关联容器

* [set](http://en.cppreference.com/w/cpp/container/set) 
* [map](http://en.cppreference.com/w/cpp/container/map)
* [multiset](http://en.cppreference.com/w/cpp/container/multiset)
* [multimap](http://en.cppreference.com/w/cpp/container/multimap)



## 无序关联容器

* [unordered_set](http://en.cppreference.com/w/cpp/container/unordered_set)
* [unordered_map](http://en.cppreference.com/w/cpp/container/unordered_map)
* [unordered_multiset](http://en.cppreference.com/w/cpp/container/unordered_multiset)
* [unordered_multimap](http://en.cppreference.com/w/cpp/container/unordered_multimap)



## Container adaptors

* [stack](http://en.cppreference.com/w/cpp/container/stack)
* [queue](http://en.cppreference.com/w/cpp/container/queue)
* [priority_queue](http://en.cppreference.com/w/cpp/container/priority_queue)




## vector

**特点：**

* 数据连续存放
* 可以动态调整大小



**API 简介：**

* `push_back`: 
  * 对于临时对象，调用对象的移动构造函数
  * 对于非临时对象，调用对象的复制构造函数
* `pop_back()` ：会析构对象
* `back()` : 返回的是引用。
* `=` : 
  * vector 的 `=` 中，调用的是里面所包含对象的复制构造函数
  * vector 的移动赋值操作中，直接移动 `vector` 内部的资源，不会调用所包含对象的任何函数





## 参考资料

[http://en.cppreference.com/w/cpp/container](http://en.cppreference.com/w/cpp/container)