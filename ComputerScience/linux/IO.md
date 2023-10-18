1. 阻塞IO，调用一些IO操作会导致block住
2. 非阻塞IO，调用不会阻塞，可以立即返回。（和多线程调用IO效果类似）。该IO对于普通文件的IO是无效的
3. IO多路转接：首先得是非阻塞描述符
    1. 假设有多个输入。哪个输入准备好了，就用哪个输入，这种场景就非常适合IO多路转接
    2. 构造一个感兴趣的文件描述符表，然后调用一个函数，直到表中的某个文件描述符准备好IO时，这个函数才会返回。当这些函数返回时，会告诉我们哪些文件描述符可以进行IO
    3. pool, pselect, select 这三个函数可以让我们能够执行IO多路转接
    4. select：主要用于终端IO和网络IO，在普通文件IO上是不是起作用？

```c
select

/*
select输入：
    1. 我们所关心的描述符
    2. 每个描述符所关系的条件（是想从一个描述符读，还是往一个描述符写，是否关心描述符的异常条件）
    3. 愿意等待多长时间（可以永远等待、等待一个固定时间、根本不等待）

当select返回时，它会告诉我们：
    1. 已准备好的描述符的总数量
    2. 对于读、写、异常这三个条件的每一个，哪些描述符已经准备好

知道哪些描述符准备好之后，就可以进行相应操作了。

注意：如果一个描述符碰到了文件尾端，那么select会认为该描述符是可读的，但是真正去读的时候 read 返回 0

对于准备好的定义：
    1. 对于读集合，如果 read 不会阻塞，那就是准备好的
    2. 对于写集合，如果 write 不会阻塞，就是准备好的
    3. 对于异常集合，如果 存在 未处理的异常，就是准备好的
    4. 对于读、写、异常条件。普通文件的文件描述符总是返回准备好（所以select对于普通文件IO是不生效的！！！）
*/

```

```c
// 返回值表示 满足事件的项数
int poll(struct pollfd fdarray[], ndfs_t nfds, int timeout);

struct {
    int fd;
    short events;  // 感兴趣的事件，由调用者设置。
    short revents; // 发生在 fd 上的事件。由内核设置
}

/*
类似select，但是程序接口不大一样.

*/

```

```c
epoll

// 创建一个 epoll 实例，翻译 epoll实例的 fd. epoll_fd
int epoll_create(int size)

//注册想要监听的 fd, op:①EPOLL_CTL_ADD, ②EPOLL_CTL_MOD，③EPOLL_CTL_DEL
int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event);

typedef union epoll_data {
    void        *ptr;
    int          fd;
    uint32_t     u32;
    uint64_t     u64;
} epoll_data_t;

struct epoll_event {
    uint32_t     events;      /* Epoll events */
    epoll_data_t data;        /* User data variable */
};

/* 事件。
EPOLLIN：fd read 不阻塞
EPOLLOUT：fd write 不阻塞
EPOLLRDHUP：Stream socket peer closed connection, or shut down writing half of connection. (This flag is especially useful for writing simple code to detect peer shutdown when using Edge Triggered monitoring.)
EPOLLPRI：There is urgent data available for read(2) operations.
EPOLLERR：Error condition happened on the associated file descriptor. epoll_wait(2) will always wait for this event; it is not necessary to set it in events.
EPOLLHUP：Hang up happened on the associated file descriptor. epoll_wait(2) will always wait for this event; it is not necessary to set it in events.
EPOLLET：Sets the Edge Triggered behavior for the associated file descriptor. The default behavior for epoll is Level Triggered. See epoll(7) for more detailed information about Edge and Level Triggered event distribution architectures.
EPOLLONESHOT：Sets the one-shot behavior for the associated file descriptor. This means that after an event is pulled out with epoll_wait(2) the associated file descriptor is internally disabled and no other events will be reported by the epoll interface. The user must call epoll_ctl() with EPOLL_CTL_MOD to rearm the file descriptor with a new event mask.
*/

// 等待事件发生，blocking the calling thread if no events are currently available.
// 有事件发生的fd 会放在 events中，events是个buffer，maxevents表明 buffer 中最多可以存放的事件个数
int epoll_wait(int epfd, struct epoll_event *events,
               int maxevents, int timeout);

The data of each returned structure will contain the same data the user set with an epoll_ctl(2) (EPOLL_CTL_ADD,EPOLL_CTL_MOD) while the events member will contain the returned event bit field.


/*
epoll 的 edge_triger 与 legel_triger

level_trigger：和 select, pool 表现一致。如果 读不阻塞，或者写不阻塞，就会触发事件
edge_trigger: 状态变换才会触发事件。初始状态是什么时候的状态？

*/
```


4. 异步IO：


# 参考资料

1. https://linux.die.net/man/7/epoll
2. 
