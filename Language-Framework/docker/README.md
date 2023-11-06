* docker：是一个 C/S 架构的工具，docker服务，用于接收服务器发来的指令，docker客户端，就是我们经常敲的 docker 命令。
* image(镜像)：image 和 container 的关系就类似 源代码 和 执行的代码的关系
* container(容器)

# 简单命令汇总

```shell
# 启动 docker 服务

sudo systemctl restart docker

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


# 数据传输
docker cp /opt/test/file.txt mycontainer:/opt/testnew/  # 宿主机->容器
docker cp mycontainer:/opt/testnew/file.txt /opt/test/  # 容器->宿主机


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
docker image rm $image_name # 删除镜像
docker search $image_name # 查找镜像

docker inspect $image_name #查看镜像的详细信息
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

--name some_name 指定容器的名字。
-p 80 指定公开哪些端口给宿主机，如果有多个，那就多个 -p

-p8080:80 宿主机8080 和 容器 80 进行绑定
--rm 退出容器后，容器被删除

-e "WEB_PORT=8080" 运行时指定容器的环境变量

--volume, -v. -v host_dir:container_dir 将主机的目录映射(挂载)到 容器内的某个目录。
```

# 构建镜像
* 使用 `docker commit` 将当前的容器状态 构建镜像
* 使用 `docker build & Dockerfile` 构建镜像


使用 docker commit构建镜像流程：
* 创建一个容器
* 在容器中进行修改（装软件，创建文件，etc）
* 将修改提交为一个新镜像

```shell
docker run -i -t ubuntu /bin/bash

# do some installation within container
# exit container

# 这个是提交到了 远程还是本地？  应该是本地吧
docker commit 4aab3ce image_repo_name/image_name

docker commit \
  -m"message" \
  -a"author" \
  4aab3ce \
  image_repo_name/image_name:tag

```


使用 docker build & Dockerfile 构建镜像
* 就像是在写配置文件(Dockerfile)， docker build 根据 Dockerfile进行构建镜像

```Dockerfile
FROM ubuntu  # 基础镜像
MAINTAINER ky "yinpenghhz@hotmail.com" # 作者信息

ENV RVM_APTH /home/rvm #镜像构建过程中 设置环境变量。该变量可以在Dockerfile中使用。$RVM_APTH. 这些环境变量也会持久化到使用该镜像创建的任何容器中。

# 以下开始执行命令
# 默认情况下，RUN指令 会在shell里使用命令 包装器 /bin/bash -c "cmd_str" 来执行。如果在不支持shell的平台，或者不希望在 shell中执行(比如避免shell的字符串篡改)，可以用 exec 格式执行
RUN apt-get update && apt-get install -y nginx
RUN echo 'Hi' 

RUN ["apt-get", "install", "-y", "nginx"] # exec格式的run指令



WORKDIR /opt/web/app #容器内部设置的一个工作目录，ENTRYPOINT/CMD/ "./" 都会在该目录下执行

VOLUME #向容器中添加卷，见 docker run 的 --volumn

ADD file.txt /container/path/to/file.txt #将构建环境下的 文件/目录 复制到镜像中。 如果 目的地址以 / 结尾，docker就认为源是一个目录，否则，认为源是文件
ADD http://wordpress.org/file.zip /root/file.zip
ADD file.zip /root/file/ # 会对其进行解压

# copy 只关心在构建上下文传文件，而且不会做 压缩、解压 操作
COPY dir/ /container/dir/ # dir目录下的文件传输

# 向镜像中添加元数据。
LABEL version="0.1.0"
LABEL location="China" type="What?"  # docker inspect image_name 查看 LABEL

# 停止容器时，发送什么系统调用信号给 容器
STOPSIGNAL SIGKILL

# docker build时，传递给 构建运行时的变量 docker build --build-arg build=1234 -t repo_name/image_name 
ARG build
ARG webapp_user=rocy

# 添加触发器, 当一个镜像被用作其它镜像的基础镜像时，该触发器会被执行。
ONBUILD ADD . /app/src


# CMD:指定容器启动时该运行什么命令
CMD /bin/bash # docker会在命令前 加 /bin/bash -c 来执行
# CMD ["/bin/bash"] # exec格式。
# docker 命令行可以覆盖 CMD指令 docker run -i -t ubuntu /bin/bash. /bin/bash会取代Dockerfile配置的CMD指令

# ENTRYPOINT 指定容器启动时 执行什么命令。不会被 docker run 的参数覆盖，docker run 中的参数 会被 传递给 ENTRYPOINT 指定的指令
ENTRYPOINT ["/usr/sbin/nginx", "-g", "daemon off;"] # exec 格式。也可以 /bin/bash 格式

# 当指定了 ENTRYPOINT 时，CMD 就变成了传给 ENTRYPOINT 的参数. ENTRYPOINT ["/usr/sbin/nginx"]; CMD ["-h"]. 那么会按照 /usr/sbin/nginx -h 启动容器。CMD 依旧可以被 命令行覆盖



EXPOSE 80
```

```shell
# 如何基于Dockerfile构建
# 1. cd 到 Dockerfile存在的目录
# 2. 执行下面命令
docker build . -t="repo_name/image_name[:tag]"

# 或者指定 Dockerfile存在的位置
docker build . -t="repo_name/image_name[:tag]" git@github.com:git_prog_path #这个假设git_prog_path这个repo里有 Dockerfile
docker build . -t="repo_name/image_name[:tag]" /path/to/your/dockerfile #这个直接指定的 dockerfile的绝对路径

# 其它指令
# --no-cache 忽略dockerfile的构建缓存
```


# 参考资料

1. https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
