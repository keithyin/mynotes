# github操作基础

## 工作区与暂存区

* 工作区：即你看到的目录就是工作区

* 暂存区：工作区中有个隐藏目录`.git`,这个不算工作区，是`git`版本库,其中最重要的就是称为`stage`的暂存区，还有一个`git`自动为我们创建的第一个分支`master`,以及指向`master`的一个指针`head`.

  ![](http://www.liaoxuefeng.com/files/attachments/001384907702917346729e9afbf4127b6dfbae9207af016000/0)





## 版本控制

* 可以很容易的比较新旧版本的区别

**How could having easy access to the entire history of a file make you a more efficient programmer in the long term?**

* git diff commit_id1 commit_id2 可以打印出两次commit之间的修改
* git log 可以打印出commit日志
* ​