```c++
#include <algorithm>
```



# 条件判断

* 都是 `[first, last)`

### `std::all_of`

* 区间内的所有的预测都为 `true`, 或者区间为空集, 返回 `true` ; 否则返回 `false`
* `[first, last)`

```c++
template<class InputIterator, class UnaryPredicate>
  bool all_of (InputIterator first, InputIterator last, UnaryPredicate pred)
{
  while (first!=last) {
    if (!pred(*first)) return false;
    ++first;
  }
  return true;
}
```

### `std::any_of`

* 区间内有任何为 `true` , 则为 `true` , 否则为 `false`, 空集也为 `false`
* `[first, last)`

```c++
template<class InputIterator, class UnaryPredicate>
  bool any_of (InputIterator first, InputIterator last, UnaryPredicate pred)
{
  while (first!=last) {
    if (pred(*first)) return true;
    ++first;
  }
  return false;
}
```

### `std::none_of`

* 区间内, 如果都为 `false` , 则返回 `true`, 空集返回 `true`



# 遍历&查找

### `std::for_each`

* 功能等价于以下代码, 对区间中的每个元素执行函数操作, 如果想 `inplace` 操作的话, `fn` 的形参可以是引用. 

```c++
template<class InputIterator, class Function>
  Function for_each(InputIterator first, InputIterator last, Function fn)
{
  while (first!=last) {
    fn (*first);
    ++first;
  }
  return fn;      // or, since C++11: return move(fn);
}
```

### `std::transform`

* 功能等价于以下代码; 对区间中的每个元素进行转换, 结果保存在另一个集合中

```c++
template <class InputIterator, class OutputIterator, class UnaryOperator>
  OutputIterator transform (InputIterator first1, InputIterator last1,
                            OutputIterator result, UnaryOperator op)
{
  while (first1 != last1) {
    *result = op(*first1);  // or: *result=binary_op(*first1,*first2++);
    ++result; ++first1;
  }
  return result;// 返回的是 end
}
```



### `std::find`

* 功能等价于以下代码, 查找序列中是否有想要的元素, 返回第一个所在的位置

```c++
template<class InputIterator, class T>
  InputIterator find (InputIterator first, InputIterator last, const T& val)
{
  while (first!=last) {
    if (*first==val) return first;
    ++first;
  }
  return last;
}
```

* 其它特征
  * 时间复杂度: 线性

### `std::find_if`

* 功能等价于以下代码, 返回第一个使`pred` 为 `true` 的位置

```c++
template<class InputIterator, class UnaryPredicate>
  InputIterator find_if (InputIterator first, InputIterator last, UnaryPredicate pred)
{
  while (first!=last) {
    if (pred(*first)) return first;
    ++first;
  }
  return last;
}
```

* 其它特征
  * 时间复杂度: 线性

### `std::find_if_not`

* 返回第一个 `pred` 为 `false` 的位置; 如果没有, 返回最后一个

### `std::find_end`

* 两个区间, 查找第二个区间在第一个区间的最后一个的第一个元素.

```c++
template<class ForwardIterator1, class ForwardIterator2>
  ForwardIterator1 find_end (ForwardIterator1 first1, ForwardIterator1 last1,
                             ForwardIterator2 first2, ForwardIterator2 last2)
{
  if (first2==last2) return last1;  // specified in C++11

  ForwardIterator1 ret = last1;

  while (first1!=last1)
  {
    ForwardIterator1 it1 = first1;
    ForwardIterator2 it2 = first2;
    while (*it1==*it2) {    // or: while (pred(*it1,*it2)) for version (2)
        ++it1; ++it2;
        if (it2==last2) { ret=first1; break; }
        if (it1==last1) return ret;
    }
    ++first1;
  }
  return ret;
}
```

### `std::find_first_of`

* 和上面的相反, 上面返回的是 最后一个, 这个返回的是第一个 的第一个元素的位置

### `std::adjacent_find`

* 区间中第一个 连续相等的 第一个的位置

```c++
template <class ForwardIterator>
   ForwardIterator adjacent_find (ForwardIterator first, ForwardIterator last)
{
  if (first != last)
  {
    ForwardIterator next=first; ++next;
    while (next != last) {
      if (*first == *next)     // or: if (pred(*first,*next)), for version (2)
        return first;
      ++first; ++next;
    }
  }
  return last;
}
```



## 二分查找

* 数据按照 **逻辑上的** 从小到大排列
* 逻辑上的小定义为 `element<value == true` 或者 `comp(element, value) == true` 

```c++
template< class ForwardIt, class T >
bool binary_search( ForwardIt first, ForwardIt last, const T& value );

template< class ForwardIt, class T, class Compare >
bool binary_search( ForwardIt first, ForwardIt last, const T& value, Compare comp );

// 注意 Compare 的签名为 bool(const T&, const T&);


// binary_search 的可能实现方式
template<class ForwardIt, class T, class Compare>
bool binary_search(ForwardIt first, ForwardIt last, const T& value, Compare comp)
{
    first = std::lower_bound(first, last, value, comp);
    // 因为是 lower_bound, 所以 *first<=value 
  	// 现在只要 value>=*first 就能说明value == *first, 
  	// !comp(value, *first) 就是 value>=*first 的含义.
    return (!(first == last) && !(comp(value, *first))); 
}
```

```c++
#include <iostream>
#include <algorithm>
#include <vector>

bool compare(const int &a, const int& b) {
    return a<b; // 如果这里改成 a>b 的话, 就不行了.
}
 
int main()
{
    std::vector<int> haystack {1, 3, 4, 5, 9};
    std::vector<int> needles {1, 2, 3};
 
    for (auto needle : needles) {
        std::cout << "Searching for " << needle << '\n';
        if (std::binary_search(haystack.begin(), haystack.end(), needle, compare)) {
            std::cout << "Found " << needle << '\n';
        } else {
            std::cout << "no dice!\n";
        }
    }
}
```







# 计数

### `std::count`

* 区间内某个值出现了几次

### `std::count_if`

* `pred` 为 `true` 的计数





# 参考资料

http://www.cplusplus.com/reference/algorithm/