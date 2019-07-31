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


## 一些重置操作
* 重置 git add .
  * git reset <filename>
* 重置 git commit
  * git reset --soft HEAD~1 ，重置 commit 到上一次commit，不删除本地修改的文件
  * git reset --hard HEAD~1 , 重置 commit 并且删除本地修改
  
## git rebase
* clone 远程的 master 分支修改代码
* 打算提交，发现别人抢先一步提交了
* 这时候需要 git pull 下来然后手动进行一些合并操作，然后正常流程是 git add, git commit, git push
* 在 git commit 之后 git rebase一下，可以使得提交过程干净一些
  
  
  
## 参考资料
[https://www.freecodecamp.org/forum/t/how-to-undo-a-git-add/13237](https://www.freecodecamp.org/forum/t/how-to-undo-a-git-add/13237)
[https://www.git-tower.com/learn/git/faq/undo-last-commit](https://www.git-tower.com/learn/git/faq/undo-last-commit)
