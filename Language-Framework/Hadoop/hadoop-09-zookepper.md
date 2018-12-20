# ZooKeeper

* 分布式协调服务，基础服务



**zookeeper本身是个集群**

* 提供少量数据的存储和管理
* 提供对数据节点的监听器
* 可用来保存 应用的配置信息



**Zookeeper  Service**

* 工作的时候有两个角色，Leader/Follower，在配置的时候不需要指定，运行的时候会通过一个选举机制选择出 Leader
* Leader：写操作都由 Leader 执行



**为什么使用 ZooKeeper**

* 大部分分布式应用需要一个主控、协调器或控制器来管理物理分布的子进程（如资源、任务分配等）
* 目前，大部分应用程序需要开发私有的协调程序，缺乏一个通用机制
* 协调程序的反复编写浪费时间，且难以形成通用、伸缩性好的协调器
* ZooKeeper：提供通用的分布式锁服务，用以协调分布式应用



**ZooKeeper 的安装与配置**

* 在 conf 目录下创建一个配置文件 zoo.cfg

```shell
tickTime=2000
dataDir=/path/zookeeper/data
dataLogDir=/path/zookeeper/dataLog
clientPort=2181
initLimit=5
syncLimit=2
server.1=server1:2888:3888
server.2=server2:2888:3888
server.3=server3:2888:3888
```

