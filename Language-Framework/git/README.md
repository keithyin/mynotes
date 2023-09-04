安装 

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
