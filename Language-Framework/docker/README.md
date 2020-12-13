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
docker stop $container_name (停止容器)
docker restart $container_name (重启容器)
docker rm $container_name # 删除容器

docker run --name $your_container_name -i -t -d $image_name /bin/bash (创建容器, 守护式容器)

# 查看容器内部的情况，对于非deamon容器，直接进去执行linux命令即可。如果是deamon容器，可以通过
docker top $daemon_container

docker exec -d $daemon_container touch /path/to/file # 执行 后台 命令，-d 表示后台命令
docker exec -t -i $daemon_container /bin/bash # 这样之后我们就可以用命令行操作 容器内部了。

# 操作镜像
docker images # 打印所有镜像
docker pull $image_name:$version # 拉取（下载）镜像
docker images $image_name # 查看 image 信息
docker search $image_name # 查找镜像
```

# 构建镜像
* 使用 `docker commit` 构建镜像
* 使用 `docker build & Dockerfile` 构建镜像
