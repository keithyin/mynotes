# Ubuntu（linux）文件系统结构

## 主目录
* /bin : 包含系统应用的地方，常用的命令，`ls` ,`rm` 等等都在这。

* /sbin : 包含一些只能被超级用户使用的 命令。`s` 应该代表的就是`super`的意思。

* /etc ： 包含系统全局配置文件的地方，影响系统的行为。

* /lib ： 包含非常重要的动态链接库和 `kernel modules` 的地方。

* /root ： 超级用户的 `home` 目录。

* /home ： 用户的 `home` 目录。

* /tmp： 应用放临时文件的地方。 

* /usr ： 包含大部分`用户`的 工具和应用，部分的复制了根目录结构，例如，包含 `/usr/bin:/usr/lib`

* /opt ： 可以用来存储不用`package manager`管理的软件。i.e. 存放不是用包管理器安装的软件的地方。

* /media ： 外部设备的挂载点，`U盘`就挂载在这上边。

* /mnt ： 也是一个挂载点，主要是用于临时挂载设备，例如网络文件系统。

* /boot ： 包含启动系统所需的文件，包含 `linux kernel，bootlader configuration files`

* /dev ： 包含所有的设备文件。

* /proc ： 一个虚拟的文件系统，提供让 `kernel` 给 `processes` 发消息的机制。 

## 参考资料
[https://help.ubuntu.com/community/LinuxFilesystemTreeOverview](https://help.ubuntu.com/community/LinuxFilesystemTreeOverview)