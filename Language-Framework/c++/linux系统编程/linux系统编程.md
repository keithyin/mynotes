# linux 系统编程

```shell
ulimit -a # 当前系统的资源上限
```



![](../..//imgs/memory-segment.png)

## 操作系统基础

* 程序: 二进制文件.
* 进程: 执行的程序. 

**PCB: 进程控制块**

* 内核空间
* 本质是一个 结构体
  * 进程 id
  * 继承的状态: 就绪, 运行, 挂起, 终止
    * 就绪: 除了 CPU 资源, 其它资源都有了
    * 挂起: 在等待其它资源
  * 进程切换时需要保存和恢复的 CPU 寄存器的值
  * 描述虚拟地址的空间的信息
  * 描述控制终端信息
  * 进程的当前工作目录
  * umask 掩码
  * 文件描述符表
  * 和信号相关的一些信息
  * 用户 id 和 组 id
  * 会话 和 进程组
  * 进程可以使用的资源上限



**环境变量**

* 操作系统运行环境的一些参数
* `PATH`: 可执行程序的搜索路径, 从前往后开始检索的
* 如果使用环境变量, 需声明环境变量 `extern char **environ`
  * 环境变量以 `NULL` 结尾
  * 命令行参数也要以 `NULL` 结尾

```c++
#include <stdlib.h>
char *getenv(const char * name);
int setenv(const char * name, const char *value, int overwrite);
int unsetenv(const char *name);
```

* `NULL` 指针指的地方是什么意思



## Linux 操作系统一些基本概念

* 终端：输入输出设备的总称，虚拟终端
  * 用户通过终端登陆系统后得到一个 `shell` 进程， 这个终端成为 `shell` 进程的控制终端
  * 进程中，控制终端是保存在 PCB 中的信息，而 fork 会复制 PCB 中的信息。所以 `shell` 进程启动的其它进程的控制终端 和 `shell` 进程同一个终端
  * 没有重定向的情况下，每个进程的标准输入，标准输出或者标准错误输出都指向控制终端
  * 在控制终端输入一些特殊的控制键可以给 **前台进程** 发信号
* linux 系统的启动
  * `init` --> `fork` -->`exec(getty)`--> 
* 网络终端：
  * 客户端 --> 服务器 ---> 伪终端主设备 ---> 伪终端从设备
* 进程组：（和作业一个概念，进程的集合）
  * 父进程创建的 子进程和自己属于同一个进程组，进程组 id 就是 父进程 id
  * 只有父进程的进程 id 和 进程 组 id 一样
  * 父进程可以从进程组中脱离出来
* 会话：一组进程组
  * 创建会话后，会话就没有控制终端了。
* 守护进程：（Daemon），后台服务进程，通常独立于终端并且周期性地执行某种任务或等待处理某些发生的事件。一般采用以d结尾的名字
  * 调用 `setsid` 函数创建一个新的 `session`， 并成为 `session leader`



## 文件

* `stat 结构体`
  * 文件设备编号
  * inode: i节点
  * 文件类型和存取权限, 16 bits, 用 8 进制写进去就行了
  * 文件的硬连接数, 刚建立的文件为 1
  * 用户 id, 组 id
  * 设备类型
  * 文件字节数, 块大小, 块数

```c
// 追踪软链接到原文件的大小
int stat(const char* path, struct stat *buf);

int fstat(int fd, struct stat *buf);

// 不追踪软链接
int lstat(const char* path, struct stat* buf);

// 判断文件的一些属性
int access(const char *path, int mode);

link();
symlink();

/*
删除一个文件的目录项并减少它的链接数, 若成功返回0,否则返回-1, 错误原因存在 errno
如果想通过调用这个函数来成功删除文件, 必须拥有这个文件所属目录的写和执行权限

如果是符号链接, 删除符号链接
如果是硬链接, 硬链接数-1, 当减为0时, 释放数据块和 inode
如果文件链接数为0 但是已有进程打开该文件,并持有文件描述符,则等该进程关闭该文件时, kernel 才会真正的删除该文件
*/
unlink();
```





## 进程控制

* 代码中创建进程, 
* `fork`: 创建一个子进程, 

```c++
#include <unistd.h>
pid_t fork(void); 
/*
返回值有两个:
	1. 返回子进程 pid
	2. 返回 0
*/
pid_t getpid(); // 获取自己 的 pid
pid_t getppid(); // 获得父进程的 pid

```



```c
#include <unistd.h>

int main(){
  int a=0;
  int b = 0;
  
  // 这个位置会创建一个子进程, !!子进程和父进程一样继续往下走, 执行过的不会再执行了
  // 父进程的 fork 返回 子进程的 id, 子进程的 fork 返回 0(表示创建成功)
  pid_t v = fork();
}
```

* **刚刚** fork 子进程之后:
  * 一样: 全局变量, .data, .text, 堆, 栈, 环境变量, 用户 id, 宿主目录, 进程工作目录, 信号处理方式
  * 不一样: 进程id, fork 的返回值, 父进程id, 进程运行时间, 定时器, 未决信号集
  * 似乎, 子进程复制了父进程的 0-3G 用户空间内容
    * 只是似乎而已: 但实际上遵循 读時共享 (共享物理地址), 写时复制 原则
* 父子进程**共享**:  文件描述符, mmap 建立的映射区.
  * 所以可以通过 文件描述符 和 mmap 建立的映射区 进行数据共享

**循环创建5个子进程**

```c
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

int main(){
    for(int i=0; i<5; i++){
        pid_t pid = fork(); // 创建子进程, 子进程复制父进程的运行状态
        if (pid>0){
            continue;
        }else if (pid==-1){
          perror("fork error");
        }else{       // 如果这儿不控制, 子进程还会创建子进程
            break;  // 创建的子进程就不要凑热闹了
        }
    }

    printf("hello\n");
    
    return 0;
}
```

* 有效用户 `id` 与 实际用户 `id`

```c

// sudo apt-get 加了个 sudo ,有效就是 root 实际还是 当前用户
uid_t getuid(void); //实际用户 id
uid_t geteuid(void); // 有效用户 id 
```



**exec 函数族**

* `fork` 创建子进程后执行的是和父进程相同的程序(但是有可能执行的是不同的代码分支), 子进程往往要调用一种 `exec` 函数以执行另一个程序. 当进程调用一种 `exec` 函数时, 该进程的用户空间代码和数据完全被新程序替换, 从新程序的启动例程开始执行. 
* 调用 `exec` 并不创建新进程, 所以调用 `exec` 之后进程 pid 不变
* 作用: 在程序中执行一个进程

```c
// 目录+程序名 操作
int execl(const char *path, const char *arg, ...);


// arg 是 arg0,
// execlp 中的 p 表示 有 PATH 参与
// execlp("ls", "ls", "-a", "-l", "-l", NULL)
int execlp(const char *file, const char *arg, ...);
```



**回收子进程**

* 孤儿进程: 父进程先于子进程结束, 则子进程成为孤儿进程, 子进程的父进程会变成 `init` 进程, 称为 `init` 进程领养孤儿进程, 由`init` 负责子进程的回收
* 僵尸进程: 子进程终止, 但是父进程尚未回收, 子进程的残留资源(PCB) 存放于内核中, 变成僵尸进程
  * 僵尸进程是不能使用 kill 掉的, 因为 kill 是杀掉进程的 , 僵尸进程意味着子进程已经结束.
  * 父进程负责对子进程资源回收
    * `wait` : 调用一次 `wait` 只能回收一个子进程!!!!!
    * `waitpid` : 此两个函数用来回收子进程



```c
/*
	1. 阻塞等待子进程推出
	2. 回收子进程残留资源, (回收PCB)
	3. 获取子进程结束状态(退出原因)
*/
pid_t wait(int *status); // status 是传出参数


/*
	1. 可以选择不阻塞: 如果死了就回收, 没死,就不管了
*/
// 回收指定pid 的子进程, 成功返回 清理掉的 pid, 失败, -1
// WNOHANG 表示非阻塞状态, 可以使用轮询方式
pid_t waitpid(pid_t pid, int *status, int options); 


// wait if exited
WIFEXITED(status);  // 为非 0, 则进程正常结束
WEXITSTATUS(status); // 如上宏为真, 使用此宏 -> 获取进程退出状态 (exit 参数)

WIFSIGNALED(status); // 为非0, 进程异常终止
WTERMSIG(status); // 如上宏为真, 使用此宏 -> 取得使进程终止的信号的编号
```



## 进程间通信 (IPC)

* 管道 (使用最简单)
  * 伪文件, 实际是 内核的缓冲区, 以环形队列形式实现的
  * 局限性:
    * 数据自己读 自己不能写
    * 数据一旦读走, 便不在管道中存在, 不可反复读取
    * 双向半双工通信方式, 数据只能在一个方向上流动
    * 只能在有血缘关系的进程间使用管道
* fifo (有名管道): 可以在无血缘关系的进程间通信
  * 需要同一个目录下的同一个文件!!!
* 信号 (开销最小)
* 共享映射区 (可以在无血缘关系进程间通信)
* 本地套接字 (最稳定)



* 还可以使用文件进行进程间通信, 父进程和子进程 的 **kernel 部分 共享同一块物理地址空间.**



**通过管道进行进程间通信**

----

```c
// 返回值, 表示是否调用成功, pipefd 一个是写的 fd, 一个是读的 fd
// pipefd[0] 表示读端, pipefd[1] 表示写端
int pipe(int pipefd[2]);
/*
	1. 打开之后, 父子进程都会有 管道的读写
	2. 这时需要规定谁读, 谁写. 
	3. 如果父进程读, 子进程写, 则父进程需要关闭 写端, 子进程关闭读端
*/
```

* 读管道:
  * 如果管道中有数据: `read` 返回实际读到的字节数
  * 管道中无数据:
    * 写端全关闭: `read` 返回 0
    * 仍有写端打开: `read` 阻塞等待
* 写管道:
  * 读端全关闭: 进程异常终止 (SIGPIPE 信号)
  * 有读端打开: 
    * 管道未满: 写数据, 返回写入字节数
    * 管道已满: 阻塞 (少见)



```c
#include <stdio.h>
#include <unistd.h>

int main(){
    int err;
    int fd[2];
    err = pipe(fd);
    if (err==-1){
        perror("open pipe error ");
        exit(1);
    }
    pid_t pid = fork();
    // father write, son read
    if (pid>0){
      	// 父进程既然写, 那就直接 关闭读端就可以了 到底关闭的是什么 ?????????????????/
        close(fd[0]); 
        write(fd[1], "hello", sizeof("hello"));
    }else{
        close(fd[1]);
        char buf[1024];
        ssize_t size = read(fd[0], buf, 1024);
        printf("length=%d, %s\r\n", size, buf);
    }
    return 0;
}
```





**通过文件进行进程间通信**

----

* 通过共享打开的文件描述符(`FILE` 结构体), 而不是数字


```c
#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>

int main(){
    pid_t pid = fork();
  	// 不是通过 pid 这个值 来共享的哦
    if (pid>0){
        int fd = open("hello.txt", O_RDWR);
        write(fd, "hello world", sizeof("hello world"));
        wait(NULL);
    }else{
        int fd = open("hello.txt", O_RDWR);
        char buf[1024];
        read(fd, buf, 2014);
        printf("%s \r\n", buf);
    }
    return 0;
}
```






**共享内存**

----

* `mmap`
  * 借助共享内存访问磁盘空间
  * 父子进程, 兄弟之间 通信
    * `MAP_SHARED`: 可以父子间通信
    * `MAP_PRIVATE`: 父子进程似有映射区
  * 无血缘关系进程
    * ​

```c
void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t offset);
// sys_err("info");
/*
addr: 文件映射到内存中的首地址,  直接传入 NULL, 由 linux 内核指定
len: 创建的映射区大小, 通过文件的大小决定
prot: 映射区权限: PROT_READ, PROT_WRITE, PROT_READ | PROT_WRITE
flags: 内存中做的修改是否反应到磁盘上, MAP_SHARED:会反应,  MAP_PRIVATE: 不会反应
fd: 文件描述符
offset: 文件开始位置偏移 一些 字节再映射, 必须是 4K 的整数倍
返回值: 成功, 返回映射区首地址；失败:返回 MAP_FAILED 宏
*/

int munmap(void *addr, size_t len); // 关闭映射区

/*
1. 不能创建大小为 0 的映射区, 所以不能用新创建的文件建立映射区
2. 首地址不能改变, 在 munmap 的时候还需要用
3. 映射区的权限需要小于等于文件的权限, 创建映射区的过程中隐含着一次对文件的读操作
4. 不能建立大于文件大小 的 映射区
5. 映射区一旦创建成功,  fd 就可以释放了, 因为可以直接通过指针操作文件了.
6. 不能对 映射区 越界操作..
*/
```



**信号**

----

> 产生信号 ---------   内核传递信号 --------------- 处理信号



* 信号的概念 (软中断)
  * 基本属性
    * **简单**, **不能携带大量信息**, **满足某个特设的条件才能发送**
    * A 给 B 发送信号, B 不管代码执行到什么位置, 都要暂停执行, 需要处理信号, 信号处理完后再继续执行.
    * 信号: 由**内核发送, 内核处理**
  * 信号四要素
    * 编号, 名称, 事件, 默认处理动作
    * 事件: 由某事件导致信号产生
    * `kill -l` 打印出来所有支持的信号
    * `man 7 signal` 查看信号的相关描述
* 产生信号的五种方式
  * 按键产生, 如 ctrl+c (SIGINT interupt), ctrl+z (SIGTSTOP), ctrl+\ (SIGQUIT)
  * 系统调用产生: 如 kill, raise
    * raise: 给当前进程发送指定信号
    * abort(): 给自己发异常终止信号
  * 软件条件产生: 
    * 定时器 alarm: 每个进程 **有且只有一个定时器** , 到时间发送 `SIGALRM`, 自然计时法
      * `unsigned int alarm(unsigned int seconds); ` 返回 0 或上次闹钟剩余的秒数， 无失败
      * 取消定时器： `alarm(0)`
      * 无论进程处于哪种状态，alarm 都计时
    * `int setitimer(int which, const struct itimerval *new_value, struct itimerval *old_value)`
  * 硬件异常产生: 非法访问内存, 除0, 内存对齐出错
  * 命令产生: kill 命令, `kill -信号编号 进程`, 对进程操作
* 信号集操作函数
  * **信号屏蔽字(阻塞信号集)**: 用来指定屏蔽(阻塞) 哪些信号的
  * **未决信号集**:  产生 --> 递达 的中间状态

```c
typedef unsigned long sigset_t; // 类型本质是 位图

int sigemptyset(sigset_t *set); // 将集合清空
int sigfillset(sigset_t *set); // 将集合全部置1
int sigaddset(sigset_t *set, int signum); // 将某位置置一
int sigdelset(sigset_t *set, int signum); // 将某位置置0

// 某个信号是否在信号集中， 是 1 还是 0
int sigismember(const sigset_t *set, int signum); 

/*
通过自己的 set 影响 阻塞信号集
how： SIG_BLOCK, SIG_UNBLOCK, 可用于设置阻塞信号，也可以用于解除阻塞
*/
int sigprocmask(int how, const sigset_t *set, sigset_t *oldset);

// 用来读未决信号集的！！！！！！！！！
int sigpending();

int main(){
  // 通过自己的 set 影响 阻塞信号集
  sigprocmask()
}
```



* 信号处理方式:
  * 执行默认动作, (SIGKILL, SIGSTOP 不能被忽略和捕捉)
    * 终止进程
    * 忽略信号
    * 终止进程并产生 core 文件, core文件在调试中会有用
    * 暂停进程
    * 继续运行进程
  * 忽略
  * 捕捉(调用户处理函数)
* 信号捕捉
  * `signal函数`
  * `sigaction函数`
* 一些基本概念
  * 产生: 信号产生
  * 递达: 递送并且到达进程
  * 未决: 产生和滴答递达之间, 主要由于 阻塞(屏蔽)  导致该状态

```c
#include <signal.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

void sig_handler(int v){
    printf("going to exit: %d \r\n", v);
    exit(0);
}

void main(){
  	// 注册信号捕捉函数
    signal(SIGALRM, sig_handler);
    alarm(1);
    while (1){

    }
}

/*
也是用来对 某个信号注册信号处理函数
*/
int sigaction(int signum, const struct sigaction *act,
              struct sigaction*oldact);
```



## 竞态条件（时序竞态）

* `pause` 函数
  * 使得调用该函数的进程自动挂起， 需要等待一个信号才能将进程唤醒。
  * 调用此函数就是为了等待信号， 信号需要注册一个捕捉处理函数

```c
/*
有个信号 发过来， 就能把他唤醒
*/
int pause(void); 

// 使用 pause 和 alarm 实现 sleep 
int sleep(){
  // 注册 信号处理 函数
  alarm(1);
  pause();
}
```



* 全局变量的异步 IO
* 可重入函数，不可重入函数
* SIGCHLD：
  * 回收子进程
* 信号传参：但是一般不使用信号进行传参
* 信号中断系统调用





## 线程

* 线程与进程关系
  *  轻量级的进程，在 `linux` 下本质依旧是进程
  *  进程：独立地址空间，拥有 PCB
  *  线程：也有 PCB，但是没有独立的地址空间（共享）
  *  Linux 下：
    * 线程：最小执行单位。（调度的最小单位）
    * 进程：资源分配的最小单位
* Linux线程实现原理
  * 线程是轻量级的进程，也有PCB，在 `linux` 下本质依旧是进程，创建线程和进程的底层函数是一样的，都调用了 `clone`
  * 从内核看，进程和线程是一样的，因为他们都有各自不同的 PCB，但是PCB中**指向内存资源的三级页表是相同的**
  * 进程可以蜕变成线程
  * 线程可看作寄存器和栈的集合
* 线程号与线程ID
  * 线程号：OS 调度的依据
  * 线程 ID：进程内区分线程的 依据
* 线程之间共享、非共享
  * 共享资源：文件描述符表，信号的处理方式，当前工作目录，用户ID和组ID，内存地址空间（`.text/.data/.bss/heap/共享库`， 除了栈不共享）, **栈上的，传个地址也能访问**。
  * 线程独享：线程id，处理器现场和栈指针，栈空间，**errno变量**，信号屏蔽字，调度优先级
* 优缺点
  * 优点：1.提高程序并发性。2.开销小。3.数据通信共享数据方便。
  * 缺点：1. 库函数，不稳定。2. 调试，编写困难，gdb不支持。3. 对信号支持不好
* 线程控制原语

```c
// 返回当前线程 id， 编译时候家 -pthread
#include <pthread.h>
pthread_t pthread_self(void);

// 创建线程 并 执行线程
// 最后一个是 线程处理函数的 参数
int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start_routine)(void*), void *arg);


// 将当前线程退出::::::::::: 在线程内调用
// 无论在主控线程中执行，还是在线程中执行 exit()是将整个进程退出
void pthread_exit(void *retval);

//  阻塞等待线程退出，获取退出状态，一次只能回收一个线程。回收 PCB
//  兄弟线程也可以互相回收，detach的线程不能回收
int pthread_join(phtead_t thread, void ** retval); // retval 和 retval 相关


// 线程分离，从状态上产生分离，线程主动和主控线程断开联系。
// 好处：不用回收 PCB 了， 没有其它作用了。
// 线程结束后，其退出状态不由其他线程获取，而直接自己自动释放。网络，多线程服务器常用
// 进程就不用回收子线程的PCB了。就不用 join 等待回收了。
int pthread_detach(pthread_t thread); // 成功0,错误：错误号

// 杀死线程, 杀死之后还需要回收 PCB
// 线程的取消并不是实时的，而有一定的延时，需要等待线程到达某个取消点，
// 取消点函数：man 7 pthreads
int pthread_cancel(pthread_t thread);
```



* 修改线程属性的方法：

```c

int pthread_attr_init(pthread_attr_t *attr);
int pthread_attr_destroy(pthread_attr_t *attr);
```



* **主控线程的退出会使得线程也会被销毁**




## 线程同步

* 同步概念
  * 协调步调，不能一起乱搞
* 互斥量（互斥锁）： `pthread_mutex_*`
  * 保护共享资源
* 读写锁：`pthread_rwlock_*`
  * 与互斥量类似，但是性能更高。特性为：**写独占，读共享，写锁优先级高**
* 条件变量：`pthread_cond_*`
  * 本身不是锁，但是可以和锁一样，造成进程阻塞。
  * 通常和 互斥锁配合使用，**给多线程提供一个会合的场所**
* 信号量：互斥量升级版，可以线程间同步，可以进程间同步
* 文件锁：

**为什么需要同步**

* CPU 会对线程进行调度，所以不同线程如果对一块共享空间进行操作，会出现奇怪的问题
* 有多个控制流程操作同一个共享资源的时候，都需要进行同步



**死锁**

* 一个线程试图对同一个互斥量加锁多次
  * 写代码的时候注意一下
* 线程1拥有 A 锁，申请 B 锁， 线程2拥有B锁，申请A锁
  * 解决方法：当拿不到所有的锁的时候，就放弃已经占有的锁



**同步需要注意**

* 锁的粒度越小越好， 粒度大会托慢进程速度



**互斥量**

* 可用在线程间，也可以使用在进程间

```c
// 互斥锁 pthread_mutex_t 类型，当成整数就可以了，就有两个值，0 和 1
/*
都是成功返回 0, 失败返回错误号
*/

// 初始化一个互斥量，初值可看成1
int pthread_mutex_init(pthread_mutex_t *restrict mutex, 
                       const pthread_mutexattr_t *restrict attr);

// 加锁：如果 mutex 是1 ，则 -1 然后访问，如果 mutex==0, 则阻塞等待锁被释放
int pthread_mutex_lock(pthread_mutex_t *mutex);

// 尝试枷锁，如果可以加，就加锁， 如果没法加，就返回另一个值，
// 可以使用轮询机制使用此函数枷锁
int pthread_mutex_trylock(pthread_mutex_t *mutex);

// 解锁
int pthread_mutex_unlock(pthread_mutex_t *mutex);

/*
restrict 关键字：只用语修饰指针，告诉编译器，所有修改本指针指向内存中的内容，只能通过本指针完成，不能通过本指针以外的其它变量或指针修改。
*/
```

```c
// 进程间进行共享，在 fork 之前将锁处理好然后用就行了。
```



**读写锁**

* 也是一把锁，但是可以选择不同的加锁方式
* 读共享， 写独占，读得时候不能写，读写一起竞争锁的话，写胜出

```c
// 写独占，读共享，写锁优先级高, pthread_rwlock_t 类型，用于定义一个锁变量
int pthread_rwlock_init(pthread_rwlock_t *restrict rwlock,
                       const pthread_rwlockattr_t *restrict attr);
int pthread_rwlock_destroy(pthread_rwlock_t *rwlock);
int pthread_rwlock_rdlock(pthread_rwlock_t *rwlock);// 以读方式加锁
int pthread_rwlock_wrlock(pthread_rwlock_t *rwlock); // 以写方式加锁
int pthread_rwlock_tryrdlock(pthread_rwlock_t *rwlock);
int pthread_rwlock_trywrlock(pthread_rwlock_t *rwlock);
int pthread_rwlock_unlock(pthread_rwlock_t *rwlock);
```

```c
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>

int a = 100;

void* run(void* rwlock){
    pthread_rwlock_rdlock((pthread_rwlock_t*)rwlock);
    printf("a:%d \n", a);
    sleep(5);
    printf("a:%d \n", a);
    pthread_rwlock_unlock((pthread_rwlock_t*)rwlock);
}

int main(){
    pthread_t thread;
    pthread_rwlock_t rwlock;
    pthread_rwlock_init(&rwlock, NULL);
    pthread_create(&thread, NULL, run, (void*)&rwlock);
    sleep(1);
    pthread_rwlock_wrlock(&rwlock);
    a = 555;
    pthread_rwlock_unlock(&rwlock);
    return 0;
}
```



**条件变量**

* 生产者消费者模型
* 由于 写独立，读共享， 所以只需要一个 mutex 来帮助实现 生产者消费者模型
* 优点： 相较于 mutex 而言，条件变量可以减少竞争
  * 生产者消费者之间需要 互斥量，消费者与消费者之间也有互斥量，但如果 汇聚 中没有数据，消费者之间竞争互斥锁是无意义的。有了条件变量以后，只有生产者完成生产，才会引起消费者之间的竞争，提高了程序效率

```c
// pthread_cond_t 类型，用语定义条件变量

int pthread_cond_init(pthread_cond_t *cond, pthread_condattr_t *cond_attr);
int pthread_cond_destroy();

/*函数做了两件事：
	1. 阻塞等待条件变量满足, 释放已掌握的互斥锁，（作为一个原子操作）
	3. 当被唤醒，解除阻塞，重新申请获取互斥锁，函数返回，（作为一个原子操作）
*/
int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex);

// 有时长的阻塞, abstime: 绝对时间，1970.01.01 作为计时元年
int pthread_cond_timedwait(pthread_cond_t  *cond,  pthread_mutex_t  *mutex,  
                           const struct timespec *abstime);

/*至少？？？唤醒一个阻塞在条件变量 cond 上的线程*/
int pthread_cond_signal(pthread_cond_t *cond);

/*唤醒所有当前阻塞在 cond 上的线程*/
int pthread_cond_broadcast(pthread_cond_t *cond);
int pthread_cond_init();

```



**信号量: 进化版的互斥锁， 互斥锁初始化为 1, 信号量可一初始化为 N**

* 可以用于线程同步，也可以用于进程同步
* 信号量的初值，决定了并行的线程个数

```c
// #include <semaphore.h>

/*
pahared: 是否能在进程间共享，
value： 初始化 为 5
*/
int sem_init(sem_t *sem, int pshared, unsigned int value);
int sem_destroy();
int sem_wait(sem_t* sem); // 当前信号量为 0 的话则阻塞， 否则信号量--，
int sem_trywait(sem_t* sem);
int sem_timedwait(sem_t* sem);
int sem_post(sem_t* sem); // 信号量++

```



**文件锁：进程间同步的方法**

* 借助 `fcntl` 实现的， `file control`

```c
F_SETLK(struct flock*) ; // 设置文件锁，相当于 trylock，非阻塞的 
F_SETLKW(struct flock*); // lock 版本， 带阻塞的
F_GETLK(struct flock*);  // 获取文件锁
```






## 系统命令小结

```shell
# 会打印出程序运行所用时间 ，
# real：实际应用时间， user：运行在用户态所用时间， sys：系统态运行的时间
# 实际时间 = 用户态+内核态+等待时间。   程序运行时的速度瓶颈在 io 上
time ./app 
```



## 疑问

* 进程通过文件共享数据的原理
* 进程通过 mmap 共享数据的原理
* 管道的读写是否互斥的
* 文件共享的时候是不是互斥的
* mmap 共享的时候是不是互斥的