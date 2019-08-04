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
* 在 git pull 的时候发现远程有提交了
* 这时候 git rebase 的基本操作是
* git fetch，git rebase origin/master
* 如果有冲突：手动合并代码，git add , git rebase --continue 就可以了
* 然后 git push

## 其它
* `git pull = git fetch + git merge`
* `git merge = merge operation + git commit`(冲突了的话需要，手动解决冲突，然后 git add, git commit)
* `git rebase = rebase operation + git commit`(冲突了的话需要，手动解决冲突，然后 git add , git rebase --continue)
  
## 参考资料
[https://www.freecodecamp.org/forum/t/how-to-undo-a-git-add/13237](https://www.freecodecamp.org/forum/t/how-to-undo-a-git-add/13237)
[https://www.git-tower.com/learn/git/faq/undo-last-commit](https://www.git-tower.com/learn/git/faq/undo-last-commit)
