# linux中常用命令介绍

* `echo "hello, world"`  打印文本

* `rm -rf  directory`    rm(remove)  -r(recursive) -f(force)

* `find`  
  * find [path]\[expression]
  * `path`:find指定要查找的目录路径 `.`表示当前路径，`/`表示根目录
  * -name : 按照文件名查找文件 (在当前及其子文件中查找) `find . -name *.jpg`

* `cut` 选取指令，选出想要的字符串

  * `cut -d '/' -f3`
  * ​