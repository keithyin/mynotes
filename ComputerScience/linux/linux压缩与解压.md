# linux压缩与解压

## tar

* 解压： 常用命令 `tar -zxvf file.tar.gz`
  * `-x` : 从`archive`中解压文件
  * `-v(verbose)`: 打印出解压信息
  *  `-z`: 使用 `gzip` 压缩的`archive`
* 压缩：常用命令 `tar -zcvf target.tar.gz file_or_dir`
  * `-c`: `compress` 表示压缩



## unzip

* 安装：`sudo apt-get install unzip` 
* 解压命令：`unzip file.zip -d destination_folder`