# linux 文件

一、什么是 file inode

这篇文章介绍的很详细，简言之，就是记录文件 一些 meta-data（创建时间，用了哪些 block 来存储） 的地方。

[http://www.opsers.org/base/one-day-the-little-learning-linux-inode-detailed.html](http://www.opsers.org/base/one-day-the-little-learning-linux-inode-detailed.html)



二、Open file descriptions

The term **open file description** is the one used by POSIX to refer to the entries in the system-wide table of open files.

**open file description** 是系统级的 打开文件 表中的一个条目

> 使用系统调用 open 时， 每一次 open 都会新建一个  open file description

[http://www.man7.org/linux/man-pages/man2/open.2.html](http://www.man7.org/linux/man-pages/man2/open.2.html)

三、file descriptor

指向 open file description 的一个句柄，int 值





# 参考资料

 https://stackoverflow.com/questions/3054442/is-it-possible-to-use-threads-to-speed-up-file-reading
 https://stackoverflow.com/questions/15477925/is-is-possible-to-write-to-different-parts-of-the-same-file-from-multiple-thread 
 https://stackoverflow.com/questions/2751734/how-do-filesystems-handle-concurrent-read-write 
 https://softwareengineering.stackexchange.com/questions/163931/is-using-a-bigger-buffer-useful 

