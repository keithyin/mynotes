# ubuntu 安装 hadoop3+



**1.准备工作**

```shell
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install ssh pdsh
sudo apt-get install vim
sudo apt-get install oracle-java8-installer

#测试java是否安装成功
java -version

# 配置JAVA环境, 先打开配置文件
vim /etc/profile

# 将以下配置添加到文件末尾
export JAVA_HOME=/usr/lib/jvm/java-8-oracle  # 这里目录要换成自己的jvm里面java的目录
export JRE_HOME=${JAVA_HOME}/jre  
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib  
export PATH=${JAVA_HOME}/bin:$PATH 
export PDSH_RCMD_TYPE=ssh
```



**安装hadoop**

```shell
# 1. hadoop 官网选择想要的版本下载 binary 文件
# 2. 解压

# 修改环境变量
sudo gedit /etc/profile

# 将以下配置添加到文件末尾
export HADOOP_HOME=/software/hadoop-3.1.1 #解压目录
export PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH

# 执行 source /etc/profile
# hadoop -version 查看是否配置好环境变量
```



**配置免密登录**

```shell
ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

ssh localhost
sudo reboot
```



**hadoop 配置**

```shell
cd /software/hadoop-3.1.1/etc/hadoop

# 修改 hadoop 环境
vim hadoop-env.sh
# 文件中将 JAVA_HOME 的注释去掉，将其设置为正确的值
```



```shell
cd /software/hadoop-3.1.1/etc/hadoop
vim core-site.xml
# 文件中添加
<configuration>
<property>
<name>fs.defaultFS</name> #文件系统
# 主节点地址，端口号，默认为 9000,不要用 localhost 哦
<value>hdfs://192.168.204.200:9000</value> #这地方需要写的是uri，namenode的地址
</property>

# 配置 hadoop 的工作目录
<property>
<name>hadoop.tmp.dir</name>
<value>/home/keith/software/hadoop-3.1.1/data</value>
</property>

</configuration>
# 保存退出
```



```shell
# 配置 hdfs

vim hdfs-site.xml
# 文件中添加
<configuration>
<property>
<name>dfs.replication</name> #文件的副本数
<value>1</value>             
</property>
</configuration>
```



```shell
# 配置map-reduce
vim maprd-site.xml

<configuration>
	<property>
 		<name>mapreduce.framework.name</name> #mapreduce程序在什么框架上跑
 		<value>yarn</value>
 	</property>
</configuration>
```



```shell
# 配置 yarn

vim yarn-site.xml

<configuration>
	<property>
		<name>yarn.resourcemanager.hostname</name> # yarn集群的老大
		<value>192.168.204.200</value>
	</property>
	
	<property> # map产生的中间结果怎么传递给reduce，采用哪种机制，yarn的从节点
		<name>yarn.nodemanager.aux-services</name>
		<value>mapreduce_shuffle</value>
	</property>
</configuration>
```



```shell
# 关闭 linux 防火墙

```



**slaves文件中配置从节点**



**启动hadoop**

```shell
cd /software/hadoop-3.1.1
mkdir logs
cd logs
hdfs namenode -format   #格式化系统
start-all.sh
```

[start-dfs.sh提示rcmd: socket: Permission denied](http://mowblog.com/start-dfs-sh%E6%8F%90%E7%A4%BArcmd-socket-permission-denied/)



# hadoop 简单介绍

* 海量数据存储（HDFS），读写操作
  * 数据存在很多节点上
* 海量数据的分析（MapReduce）运算框架
  * 写代码，打包成 jar包
* 资源管理调度（YARN）
  * 分发jar包到各个节点上
  * 负责给程序分配资源

**将 运算逻辑 分发到 数据存储系统 上去运行**

**hadoop 擅长海量离线日志分析**



# 怎么解决海量数据的存储

* 大量文件
* 每个文件都非常大



**解决方法**

* 将大文件分块（blocks），不同的块分到不同的机器上去
  * 解决放在一台机器上的负载过大问题？
* 每个块会有多个副本，存在不同的机器上
  * 一旦有一台机器坏掉了，也无大碍
  * 也有利于并发访问



**Datanodes**

* 存储数据块的节点（机器）
* 被切分的数据块存放在 Datanodes 上



**Namenode**

* 一个文件有哪些切块，这些切块在哪些主机上。由Namenode管理
* 由于客户端不知道 数据到底存放在那个 Datanode 上
* 将虚拟路径 映射到 真实的路径
* 然后客户端拿到真实的 Datanodes 路径再去操作





# 海量数据计算的思路

> 在存放数据的地方计算，将计算逻辑分发到 Datanodes 上

求和：1+2+3+4+5+6+7+8+9

Map： 1+2+3， 4+5+6， 7+8+9 (在存放数据的地方计算)

Reduce：13+16+14（只在一台机器上运行，Map的运算结果通过网络传到Reduce的机器上）（Reduce也可以有多个）



**启动 Hadoop**

```shell
jps #用来查看启动了哪些服务
```



可以使用一个简单命令启动 `start-all.sh`

也可以：

* 先启动 dfs `start-dfs.sh`
* 再启动 yarn, `start-yarn.sh`



**HDFS 文件的上传与下载**

```shell
# 上传,这就将数据放在了分布式文件系统里了
hadoop fs -put filepath hsfs://url:9000/filepath

#下载
hadoop fs -get hdfs://url:9000/filepath

# 创建目录
hadoop fs -mkdir /dir1/dir2

# 查看文件
hadoop fs -cat filepath

# 9870端口可以通过 web 访问查看 hdfs 的信息
```

```python
# HDFS的基本实现思想
# 1. hdfs 通过分布式集群来存储文件，同时为客户端提供一个便捷的访问方式（虚拟地址）
# 2. 文件存储到 hdfs 集群中去的时候是被切分成 block 的
# 3. 文件的 block 存放在若干台 datanodes 节点上
# 4. hdfs 文件系统中的文件与真实的 block 之间有映射关系，由 namenode 管理
# 5. 每一个 block 在集群中会存储多个副本，好处是可以提高数据的可靠性，可以提高并发性
```



**HDFS Shell 操作**

```shell
# 注意：hdfs 中的文件是不能修改的，但是可以将文件追加进去
hadoop fs 点击回车就能出来命令
```





**执行 mapreduce 程序**

```shell
# java 写的 mapreduce 程序是要打成 jar 包的，然后再执行
hadoop jar name.jar main_class para1 para2 
```

