# BAGS, QUEUES, STACKS

* Stack （LIFO）
  * 实现方式：链表，数组，
  * C++：**stack**s are implemented as *container adaptors*, which are classes that use an encapsulated object of a specific container class as its *underlying container*, providing a specific set of member functions to access its elements. 包含 empty，size，back，push_back, pop_back 方法的 container 都可以作为 stack 的内部 container
  * https://en.cppreference.com/w/cpp/container/stack
  * resizing array vs linked list
    * linked list: 每个操作都需要 constant time，需要额外的时间和空间处理 link
    * resizing array：每个操作 constant amortized time。更少的空间浪费。
* Queue： FIFO
  * 实现方式：链表，resizing array
  * 使用 resizing array 实现的时候，最好使用循环队列。处理好 队列满的情况， 当 resizing array 满的时候， 重新分配空间， 然后将之前的值copy到新的空间上， 然后继续分配。



# String



# 图

## dynamic Connecity

问题：Given N objects（用0~N-1来建模这些objects）

两个操作：

* Union：连接两个物体。
* Find/Connected Query：判断两个objects之间有没有连接上。

algo：Quick Find （too slow）

* Integer Array of size N
* p and q connected iff they have the same id。

algo：Quick Union

* Integer Array of size N：
  * 这个 array 解释成为 一个 tree