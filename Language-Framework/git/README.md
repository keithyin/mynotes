安装 

下载源码：`https://mirrors.edge.kernel.org/pub/software/scm/git/`

centos
```shell
sudo yum remove git
sudo yum install -y xmlto
yum install curl-devel expat-devel gettext-devel openssl-devel zlib-devel
sudo make prefix=/usr/local install

```


ubuntu
```shell
apt remove git
apt-get install libcurl4-gnutls-dev libexpat1-dev gettext libz-dev libssl-dev

sudo make prefix=/usr/local install
```

安装 substree
```shell
cd git/contrib/subtree
make
sudo make install
```
