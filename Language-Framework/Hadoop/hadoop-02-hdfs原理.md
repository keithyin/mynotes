# NameNode 元数据管理机制

* HDFS 是一个集群
* 它提供了一个 虚拟的目录结构 可供 client 访问 (hdfs://ip:9000/path)
* 但实际上是分布式的存放在 datanodes 上的



**当写一个文件时，底层干了这些事**

```shell
hadoop fs -put file hdfs://ip:port/path
# 1. 客户端 先找 NameNode
# 2. NameNode 返回是否可以存，并且告诉客户端应该存在哪些 DataNodes 上
# 3. 客户端 对数据进行 切片 然后存在对应的 DataNodes 上
# 关于多副本情况，对于客户端来说，只写一个副本就够了，剩下的副本由 DataNodes 进行操作
```



**当读一个文件是，底层干了这些事**

```shell
hadoop fs -get hdfs://ip:port/path
# 
```



**NameNode 如何管理元数据**

* NN的职责
  * 维护元数据信息，虚拟路径与真实路径的映射 etc
  * 维护 hdfs 的目录结构，提供虚拟目录结构访问
  * 响应客户端的请求
* 磁盘中有一份元数据
* 内存中负责查询

```shell
# 1. 客户端上传文件时， NameNode 首先往 edits_log 文件中记录操作日志
# 2. 客户端开始上传文件，完成后返回信息给 NameNode，NN就在内存中写入这次操作日志
# 3. 在 edits_log 满的时候，将这一段时间新的元数据刷到 fsimage 文件中（edits_log 与 fsimage文件合并）
# 4. 合并操作由 SecondaryNameNode 操作
```

```shell
# SecondaryNameNode 进行合并的流程
# NameNode：当 edits_log 满的时候通知 SecondaryNN 进行 checkpoint 操作
# SecondaryNN： 通知 NameNode 停止向 edits_log 中写数据
# NameNode：创建一个新文件 edits_new
# SecondaryNN: 从NN 中下载 fsimage 和 edits_log
# SecondaryNN: 进行合并，文件名为 fsimage.checkpoint
# SecondaryNN：将合并的文件上传给 NN
```

* edits_logs, fsimage 在 namenode 的机器上的 工作文件夹内
* `cd dfs/name/current` 中有 edits_log 和 fsimage



**DataNode 工作原理**

* 提供真实数据的存储服务
  * 文件块大小：
  * 副本数：