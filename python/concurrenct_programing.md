# python 中的并发编程



## 多线程

关于多线程，应该考虑的功能有：

* 如何启动线程
* 如何关闭线程
* 如何判断线程是否还工作
* 线程间如何同步




## 创建与启动线程与终止线程

```python
from threading import Thread
# 创建线程
t = Thread(target=func, args=(arg,))
# 启动线程
t.start()

# 判断线程是否还在执行
t.is_alive()

# 等待线程结束
t.join()
```

**daemon 线程**

> 一直不断运行的 后台程序

* daemon 线程无法被连接（主线程结束后会自动被销毁掉）



**如何终止线程**

* 在某个指定的点上 轮询 *退出状态*



## 线程之间如何同步

* 使用 Event  （只用于一次性事件，i.e. 只用一次）
* 使用 Semaphore
* 使用 Condition

```python
from threading import Thread, Event
def func(started_env):
    # set 事件
    started_env.set()
    
env = Envent()
t = Thread(target=func, args=(env,))
t.start()

# 等待事件被 set
env.wait() 

```

```python
# 线程打算一遍又一遍的重复通知某个事件 用 Condition
from threading import Condition
cond = Condition()

# 通知
cond.notify_all()

# 等待
cond.wait()
```

```python
# 唤醒单独的线程 Condition，Semaphore
from threading import Semaphore
sema = Semaphore(0)

# 释放资源
sema.release()

# 等待获取资源
sema.acquire()
```



## 如何对临界区加锁

* 用 Lock 解决

```python
from threading import Lock

lock = Lock()
with lock:
    ... do something
    it is safe
    
    
# 不用 with block 的话
# lock.acquire()
# lock.release()
```



## 保存线程的专有状态

* 使用 threading.local(), 它创建了一个线程 **本地存储对象**，在这个线程上保存和读取的属性只对当前运行的线程可见，其它线程不可见

```python
import threading

def func(local_obj):
    # 只对当前线程可见
    local_obj.val = 3

local_obj = threading.local()

```



## 线程池

*  from concurrent.futures import ThreadPoolExecutor

```python
from concurrent.futures import ThreadPoolExecutor

# 创建一个可以放 n 个线程的线程池
n = 10
pool = ThreadPoolExecutor(n)

# 向 线程池提交任务
a = pool.submit(func, ...)

# 拿结果， func 返回值
x = a.result()
```



## 多进程

```python
import multiprocessing

p = multiprocessing.Process(target=func)
p.start()
...
```



```python
from concurrent.futures import ProcessPoolExecutor

pool = ProcessPoolExecutor(n)
a= pool.submit(func, ...)
```

