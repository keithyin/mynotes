# 谈谈 ubuntu 中的 /etc/profile, /etc/bashrc, ~/.bashrc, ~/.profile



**/etc/profile**  （文件）
`/etc/profile` 文件中存放的是`system-wide` 环境变量设置。一般是用来初始化一些环境变量，例如 `PATH` 或者 `PS1`

**/etc/profile.d**  （文件夹）
`/etc/profile` 在执行的时候，会会执行 `/etc/profile.d/` 文件夹下的 `*.sh` 文件，所以如果我们想要设置一些 `system-wide` 环境变量时，**推荐** 将想要设置的环境变量写到一个 `.sh` 文件中，然后放到 `/etc/profile.d/` 文件夹下。

**/etc/bash.bashrc** （文件）
这个文件用来设置 `system-wide` 的 `bash shell` 用户使用的命令别名和函数。`/etc/profile` 内部会调用这个文件。

**~/.bashrc**
`user-wide` 的配置文件

**~/.profile**
`user-wide` 的配置文件


要想搞清楚上面文件在什么时候调用，需要搞清两个概念
**Login-Shell 与 Non-Login-Shell**
* `login-shell` : 登陆系统的 `shell`
* `non-login-shell` :  登陆后打开的 `shell`，就像是 ubuntu 图形界面中 打开的 `terminal`




**Interactive-shell 与 Non-Interactive-shell**

* `interactive-shell` : 和用户存在交互的 `shell`
* `non-interactive-shell` : 和用户不存在交互的 `shell`， 比如执行程序的 `shell` 就是 `non-interactive-shell`



## 参考资料

[http://bencane.com/2013/09/16/understanding-a-little-more-about-etcprofile-and-etcbashrc/](http://bencane.com/2013/09/16/understanding-a-little-more-about-etcprofile-and-etcbashrc/)
[http://stefaanlippens.net/bashrc_and_others/](http://stefaanlippens.net/bashrc_and_others/)
[https://askubuntu.com/questions/939736/what-is-the-difference-between-etc-profile-and-bashrc](https://askubuntu.com/questions/939736/what-is-the-difference-between-etc-profile-and-bashrc)

[https://www.tldp.org/LDP/abs/html/intandnonint.html](https://www.tldp.org/LDP/abs/html/intandnonint.html)

