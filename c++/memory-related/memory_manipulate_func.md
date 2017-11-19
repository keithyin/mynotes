# C 中内存相关的一些操作



## 分配内存与释放内存

> 在C 中，下列函数的声明在 `stdlib.h` 中，在 C++， 声明在 `cstdlib` 中

* malloc
* realloc
* calloc
* free ， 释放分配的内存（分配的内存块的头部记录了当前分配内存块得大小）

### 一、 `malloc`

> malloc（memory allocate）在堆上分配内存， 分配内存块

函数原型为：

```c
void* malloc(size_t size);
```

* `size` : 所需内存字节数
* 如果成功，返回内存第一个字节的地址。如果不成功，返回空指针！



[http://www.cplusplus.com/reference/cstdlib/malloc/](http://www.cplusplus.com/reference/cstdlib/malloc/)



### 二、`realloc`

> re-allocate， 重新分配内存块

函数原型为：

```c
void* realloc (void* ptr, size_t size);
```

* 改变 `ptr` 指向的内存块的大小， 此函数可能会**移动内存块**到一个新的位置。
* `ptr` ，指向被分配的 内存块
* `size` ， 目标内存块大小

[http://www.cplusplus.com/reference/cstdlib/realloc/](http://www.cplusplus.com/reference/cstdlib/realloc/)



### 三、`calloc`

函数原型为：

```c
void* calloc (size_t num, size_t size);
```

* Allocate and zero-initialize array
* `num` , 元素的个数
* `size`， 每个元素的大小
* 相当于 分配了  `num*size` 个 字节，并初始化为 0

[http://www.cplusplus.com/reference/cstdlib/calloc/](http://www.cplusplus.com/reference/cstdlib/calloc/)



### 四、`free`

函数原型为：

```c
void free (void* ptr);
```

* 回收分配的内存块
* `ptr` ， 内存块的第一个字节的地址（malloc，realloc，calloc 的返回值）






## 内存间数据的移动和复制

> memcpy和 memmove 在头文件 `string.h` （C）/ `cstring`(C++)中，

### 一、 `memcpy`

函数原型：

```c
void * memcpy ( void * destination, const void * source, size_t num );
```

* 从 source 中 复制 num 个字节到 destination 中。
* 不会对 destination 和 source 做越界检查



### 二、 memmove

函数原型：

```c
void * memmove ( void * destination, const void * source, size_t num );
```

* 从 source 中 复制 num 个字节到 destination 中。
* 会先拷贝到一个 buffer 中，然后再到 destination 中， destination 和 source 可以 overlap





