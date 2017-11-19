# pytorch 中进程之间数据共享的方法

## win32

[http://www.cnblogs.com/henryzc/archive/2005/11/08/271920.html](http://www.cnblogs.com/henryzc/archive/2005/11/08/271920.html)

[http://bbs.csdn.net/topics/310146071](http://bbs.csdn.net/topics/310146071)

[http://blog.csdn.net/fightforprogrammer/article/details/39138005](http://blog.csdn.net/fightforprogrammer/article/details/39138005)



## linux

[http://www.cnblogs.com/timest/archive/2012/05/07/2486975.html](http://www.cnblogs.com/timest/archive/2012/05/07/2486975.html)

[http://www.linuxidc.com/Linux/2016-10/136542.htm](http://www.linuxidc.com/Linux/2016-10/136542.htm)

[mmap http://blog.csdn.net/maverick1990/article/details/48050975](http://blog.csdn.net/maverick1990/article/details/48050975)

[https://www.ibm.com/developerworks/cn/linux/l-ipc/part5/index1.html](https://www.ibm.com/developerworks/cn/linux/l-ipc/part5/index1.html)

[https://www.ibm.com/developerworks/cn/linux/l-ipc/part5/index2.html?ca=drs-](https://www.ibm.com/developerworks/cn/linux/l-ipc/part5/index2.html?ca=drs-)

## linux mmap

> mmap()系统调用 使得进程之间通过映射同一个普通文件实现共享内存。普通文件被映射到进程地址空间后，进程可以向**访问普通内存**一样对文件进行访问，不必再调用read()，write（）等操作。



### 1、mmap()系统调用形式如下：

`void* mmap ( void * addr , size_t len , int prot , int flags , int fd , off_t offset ) `

* `fd`：  为即将映射到进程空间的文件描述字，一般由open()返回。fd也可以指定为-1，此时须指定flags参数中的MAP_ANON，表明进行的是匿名映射（不涉及具体的文件名，避免了文件的创建及打开，很显然只能用于具有亲缘关系的进程间通信）。
* `len`： 是映射到调用进程地址空间的字节数，它从被映射文件开头offset个字节开始算起。
* `prot`：  参数指定共享内存的访问权限。可取如下几个值的或：PROT_READ（可读） , PROT_WRITE （可写）, PROT_EXEC （可执行）, PROT_NONE（不可访问）。
* `flags`： 由以下几个常值指定：MAP_SHARED , MAP_PRIVATE , MAP_FIXED，其中，MAP_SHARED , MAP_PRIVATE必选其一，而MAP_FIXED则不推荐使用。
* `offset`： 参数一般设为0，表示从文件头开始映射。
* `addr`： 指定文件应被映射到进程空间的起始地址，**一般被指定一个空指针**，此时选择起始地址的任务留给内核来完成。
* 函数的返回值为最后文件映射到进程空间的地址，进程可直接操作起始地址为该值的有效地址。这里不再详细介绍mmap()的参数，读者可参考mmap()手册页获得进一步的信息。



### 2、系统调用mmap()用于共享内存的两种方式：

* 使用普通文件提供的内存映射
* 使用特殊文件提供匿名内存映射



（1）使用普通文件提供的内存映射：适用于任何进程之间； 此时，需要打开或创建一个文件，然后再调用mmap()；典型调用代码如下：

```c++
int fd=open(name, flag, mode);
if (fd < 0)
  // bad
  ;
void *ptr=mmap(NULL, len , PROT_READ|PROT_WRITE, MAP_SHARED , fd , 0);
```



 通过mmap()实现共享内存的通信方式有许多特点和要注意的地方，我们将在范例中进行具体说明。



（2）使用特殊文件提供匿名内存映射：适用于具有亲缘关系的进程之间； 由于父子进程特殊的亲缘关系，在父进程中先调用mmap()，然后调用fork()。那么在调用fork()之后，子进程继承父进程匿名映射后的地址空间，同样也继承mmap()返回的地址，这样，父子进程就可以通过映射区域进行通信了。注意，这里不是一般的继承关系。一般来说，子进程单独维护从父进程继承下来的一些变量。而mmap()返回的地址，却由父子进程共同维护。 
对于具有亲缘关系的进程实现共享内存最好的方式应该是采用匿名内存映射的方式。此时，不必指定具体的文件，只要设置相应的标志即可，参见范例2。



### 3、系统调用munmap()

`int munmap( void * addr, size_t len ) `

* 该调用在进程地址空间中**解除**一个映射关系.
* `addr`: 是调用mmap()时返回的地址，len是映射区的大小。
* 当映射关系解除后，对原来映射地址的访问将导致段错误发生。

### 4、系统调用msync()

`int msync ( void * addr , size_t len, int flags) `

* 一般说来，**进程在映射空间的对共享内容的改变并不直接写回到磁盘文件中**，
* 往往在调用`munmap()`后才执行该操作。
* 可以通过调用 `msync()` 实现磁盘上文件内容与共享内存区的内容一致。



## mmap()范例

下面将给出使用mmap()的两个范例：

* 范例1给出两个进程通过映射普通文件实现共享内存通信；
* 范例2给出父子进程通过匿名映射实现共享内存。
* 系统调用mmap()有许多有趣的地方，下面是通过mmap（）映射普通文件实现进程间的通信的范例，我们通过该范例来说明mmap()实现共享内存的特点及注意事项。

### 范例1：两个进程通过映射普通文件实现共享内存通信

范例1包含两个子程序：`map_normalfile1.c` 及 `map_normalfile2.c` 。

* 编译两个程序，可执行文件分别为map_normalfile1及map_normalfile2。
* 两个程序通过命令行参数指定同一个文件来实现共享内存方式的进程间通信。
* map_normalfile2试图打开命令行参数指定的一个普通文件，把该文件映射到进程的地址空间，并对映射后的地址空间进行写操作。
* map_normalfile1把命令行参数指定的文件映射到进程地址空间，然后对映射后的地址空间执行读操作。
* 这样，两个进程通过命令行参数指定同一个文件来实现共享内存方式的进程间通信。

