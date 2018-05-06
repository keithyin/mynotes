# python 线程
线程的核心特征是它们能够以非确定的形式独立执行（即，何时开始执行、何时被打断、何时恢复执行，完全由操作系统来调度管理，这是用户和程序员都无法确定的）。

## 创建线程
`python`有个`threading`库，它可以在单独的线程中执行任意的`python`可调用对象（`函数`等等..）。
例：
```python
import threading

def run(num):
  for i in range(num):
    print(i)

#新开一个线程，执行run()
t = threading.Thread(target=run, args=(10,))#创建线程
t.start() #启动线程

```
## 线程同步问题
线程同步，即：某个线程的执行需要其它线程的配合。线程同步问题可以通过`threading.Event()`或者`threading.Condition()`来解决。
```python
import threading
import time
def run(started_evt):
  time.sleep(4)
  print("you can go")
  started_evt.set()

started_evt = threading.Event()

t = threading.Thread(target=run, args=(started_evt,))
t.start()

started_evt.wait() # wait for set

print("bye")

```
`Event`最好只用于一次性事件，即`set`后就抛弃`Event`。如果需要多次执行的话，使用`Condition`

```python
from threading import Thread, Condition

def run(n,cond):
    with cond:
        cond.wait()
        print("hello:", n)

cond = Condition()

for i in range(10):
    Thread(target=run, args=(i, cond)).start()

with cond:
    cond.notify_all()
```

## 线程间通信
将数据从一个线程发送到另一个线程，最安全的做法就是使用`queue`模块中的`Queue`，要做到这些

1. 创建一个`Queue`实例
2. 这个实例被所有线程共享
3. 线程可以通过`put()`和`get()`操作来对队列进行操作。

**Queue实例已经拥有了所有所需的锁，因此它可以安全的在任意多的线程中共享**
```python
from queue import Queue
def producer(in_q):
  while True:
    data = process_data()
    in_q.put(data)
def consumer(out_q):
  while True:
    data = out_q.get()

q = Queue()
t1 = Thread(target=producer, args=(q,))
t2 = Thread(target=consumer, args=(q,))
t1.start()
t2.start()

```

## 如何对临界区加锁
如果想要可变对象安全的使用在多线程环境中，可以使用`threading.Lock`对象来解决（对临界区加锁）。

```python

def run(val ,lock):
  with lock:
    val = val+1

lock = threading.Lock()
t = threading.Thread(target=run, args=(val, lock))
t.start()

```
