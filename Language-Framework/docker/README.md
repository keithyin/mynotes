* docker：是一个 C/S 架构的工具，docker服务，用于接收服务器发来的指令，docker客户端，就是我们经常敲的 docker 命令。
* image(镜像)：image 和 container 的关系就类似 源代码 和 执行的代码的关系
* container(容器)

# 简单命令汇总

```shell
# 启动 docker 服务

# docker 状态信息查询
docker info （查看容器，image信息）
docker ps -a (查看当前系统的容器列表)


# 容器操作
docker run --name $your_container_name -i -t $image_name /bin/bash (创建容器)
docker start $container_name (启动容器)
docker attach $container_name (重新附着到容器的会话上)
docker stop $container_name (停止容器)
docker restart $container_name (重启容器)
docker rm $container_name # 删除容器

docker run --name $your_container_name -i -t -d $image_name /bin/bash (创建容器, 交互式容器)

# 查看容器内部的情况，对于非deamon容器，直接进去执行linux命令即可。如果是deamon容器，可以通过
docker top $daemon_container  （查看容器内进程）
docker stat $deamon_container (统计信息)
docker logs $deamon_container (获取容器内的日志)
docker logs -f $deamon_container
docker logs --tail 0 -f $deamon_container
docker logs --tail 10 $deamon_container
docker logs -ft $deamon_container (跟踪最新日志)

# 容器内启动新 进程！  非deamon也可以这么用吗？感觉也可以
docker exec -d $daemon_container touch /path/to/file # 执行 后台 命令，-d 表示后台执行
docker exec -t -i $daemon_container /bin/bash # 这样之后我们就可以用命令行操作 容器内部了。

# 自动重启容器

```
--restart=always
--restart=on-failure:5
```

# 操作镜像
docker images # 打印所有镜像
docker pull $image_name:$version # 拉取（下载）镜像
docker images $image_name # 查看 image 信息
docker search $image_name # 查找镜像
```

* 交互式容器：会打开交互式shell，shell退出后，容器会停止运行

* 守护式容器： **不会** 打开交互式shell
```shell
# 打开守护容器，里面用 bash 执行 while true
docker run --name deamon_dave -d ubuntu /bin/bash -c "while true; do echo hello world; sleep 1; done"
```

## docker run

```
# docker run 命令行参数汇总

#交互式shell。docker run -i -t image_name /bin/bash
-i : 标识容器中的 stdin 是开启的
-t : 为创建的容器分配一个 伪tty终端
/bin/bash: 终端执行 该命令, 启动了一个 bash shell，也可以启动其它shell

# 

```

# 构建镜像
* 使用 `docker commit` 构建镜像
* 使用 `docker build & Dockerfile` 构建镜像


使用 docker commit构建镜像流程：
* 创建一个容器
* 在容器中进行修改（装软件，创建文件，etc）
* 将修改提交为一个新镜像

使用 docker build & Dockerfile 构建镜像
* 就像是在写配置文件(Dockerfile)， docker build 根据 Dockerfile进行构建镜像
