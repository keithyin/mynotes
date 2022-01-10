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







## 参考资料

[https://docs.python.org/2/howto/logging.html#logging-basic-tutorial](https://docs.python.org/2/howto/logging.html#logging-basic-tutorial)
