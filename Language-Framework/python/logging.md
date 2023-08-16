# Logging

## 基本组件

* Logger
  * 记录 log
* Handler
  * 将记录的 log 发送到指定的位置：标准输出，文件，标准错误输出
* Filter
  * 过滤 log，选择需要的 log 输出
* Formatter
  * 指定输出时候的格式

## 基本概念

* logging 是由 logger 的实例完成的
* logger 的每个实例都有一个名字
* logger 的名字是由 命名空间的 层次来组织的
  * 起名字的时候一般按层次起：`name="a.b.c"`
* 默认情况下，logger 并没有设置 输出destination，可以通过 `basicConfig()` 来设置。
* If you call the functions [`debug()`](https://docs.python.org/2/library/logging.html#logging.debug), [`info()`](https://docs.python.org/2/library/logging.html#logging.info), [`warning()`](https://docs.python.org/2/library/logging.html#logging.warning), [`error()`](https://docs.python.org/2/library/logging.html#logging.error) and [`critical()`](https://docs.python.org/2/library/logging.html#logging.critical), they will check to see if no destination is set; and if one is not set, they will set a destination of the console (`sys.stderr`) and a default format for the displayed message before delegating to the root logger to do the actual message output.



## Loggers

**logger 干三件事，（按执行顺序排列）**

* 运行时记录日志（根据 message创建 LogRecord）
* 根据 severity 和 filter 决定怎么处理 log（不理会或者继续往下走）
* 将 log messages 发送给 log handles



**获取Logger实例**

* `getLogger()` : 如提供名字，返回一个带名字的实例，如果不提供返回 root logger
  * 如果名字相同，返回是同一个 logger 实例
  * This name is a **dot-separated hierarchical name**, such as "a", "a.b", "a.b.c" or similar
  * ​



**Logger 的方法可以分为两类：**

* 配置logger

```python
# 指定logger所能处理的最低级别的 log，低于此级别的将被logger忽略
Logger.setLevel()

# 为 logger 添加 handler
Logger.addHandler()
# 从 logger 移除 handler
Logger.removeHandler()

# 为 logger 添加/移除 filter
Logger.addFilter()
Logger.removeFilter()
```

* 发送 log: 当logger 配置好以后，就可以创建 `log messages` 了

```python
# level 从低到高
Logger.debug()
Logger.info()
Logger.warning()
Logger.error()
Logger.critical()
```



## Filter



## Handler

> 根据 log message 的 严重性级别分发到 handler 指定的 目的地（标准输出，标准错误输出，文件，邮件）



**Handler的一些方法**

* `setLevel()`
  * 指定此 handler 处理哪个级别的 log message
* `setFormatter()`
  * 设置 Formatter 对象
* `addFilter(), removeFilter()`
  * 添加 Filter 对象或者移除 Filter对象



**Formatter**

* log message 输出前的最后一步：格式化

```python
# 两个参数，message format string, date format string
logging.Formatter.__init__(fmt=None, detefmt=None)

# fmt = '%(asctime)s - %(levelname)s - %(message)s'
# datefmt = '%m/%d/%Y %I:%M:%S %p'


logging.basicConfig(level=logging.INFO,
                    datefmt="%Y/%m/%d %H:%M:%S",
                    format="%(asctime)s - %(levelname)s - %(message)s")
```


# 多进程 logger

基本思路：
1. worker进程的logger通过配置 QueyeHandler 将 日志信息发送到 queue中
2. logging_process 读 queue 中的数据，然后打日志。


```python

# SuperFastPython.com
# example of logging from multiple processes in a process-safe manner
from random import random
from time import sleep
from multiprocessing import current_process
from multiprocessing import Process
from multiprocessing import Queue
from logging.handlers import QueueHandler
import logging
 
# 在多进程打印的时候，如果执行了下面这句。会出现 重复打印的现象！！！ 所以多进程条件下，只有 queue 进行打印就可以了
# 而且 多进程条件下，不要再调用 logging.info 这种了
# 所以 多进程 log 的最佳实践是。  就只用 QueueHandler记日志就好了。不要用 logging.info 了
# logging.basicConfig(
#                     level=logging.DEBUG,
#                     format="%(levelname)s %(name)s %(message)s %(asctime)s")

# executed in a process that performs logging
def logger_process(queue):
    # create a logger
    logger = logging.getLogger('app')
    
    # configure a stream handler
    handler = logging.StreamHandler()
    # 这里 format 是共享的！
    handler.setFormatter(logging.Formatter(fmt='%(asctime)s  %(levelname)s %(name)s %(message)s'))
    logger.addHandler(handler)
    # log all messages, debug and up
    logger.setLevel(logging.INFO)  # 这里配置的level不影响 子进程的 level

    logger.info("logger process")

    # run forever
    while True:
        # consume a log message, block until one arrives
        message = queue.get()
        # check for shutdown
        if message is None:
            break
        # log the message
        logger.handle(message)
 
# task to be executed in child processes
def task(queue):
    process = current_process()
    # create a logger
    # logger = logging.getLogger(f"app.{__name__}.{process.ident}")
    logger = logging.getLogger("app")   # 各个subprocess 这个 logger 是独立的！！
    # add a handler that uses the shared queue
    logger.addHandler(QueueHandler(queue))
    # log all messages, debug and up
    logger.setLevel(logging.DEBUG)
    # get the current process
    # report initial message
    logger.info(f'Child {process.name} starting.')
    # simulate doing work
    for i in range(5):
        # report a message
        logger.debug(f'Child {process.name} step {i}.')
        # block
        sleep(random())
    # report final message
    logger.info(f'Child {process.name} done.')
 
# protect the entry point
if __name__ == '__main__':
    logging.info("process start") # 没有配置 basicConfig是打印不出来的，如果配置了basicConfig，在多进程条件下，会有重复打印！

    # create the shared queue
    queue = Queue()
    # create a logger
    logger = logging.getLogger('app')
    # add a handler that uses the shared queue
    logger.addHandler(QueueHandler(queue))
    # log all messages, debug and up
    logger.setLevel(logging.DEBUG)
    # start the logger process
    logger_p = Process(target=logger_process, args=(queue,))
    logger_p.start()
    # report initial message
    logger.info('Main process started.')
    # configure child processes
    processes = [Process(target=task, args=(queue,)) for i in range(5)]
    # start child processes
    for process in processes:
        process.start()
    # wait for child processes to finish
    for process in processes:
        process.join()
    # report final message
    logger.info('Main process done.')
    # shutdown the queue correctly
    logging.info("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")  # 没有配置 basicConfig 是打印不出来的，如果配置了basicConfig，在多进程条件下，会有重复打印！
    queue.put(None)
    logger_p.join()
```


## 参考资料

[https://docs.python.org/2/howto/logging.html#logging-basic-tutorial](https://docs.python.org/2/howto/logging.html#logging-basic-tutorial)
