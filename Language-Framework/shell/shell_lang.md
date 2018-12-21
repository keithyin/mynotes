# shell

**开头一定要写**

```shell
#!/bin/sh
```

**Demo**
```shell
#!/bin/sh
#file : example.sh
#note : this is demo.
#date : 2018.12.21
#author : keithyin
#update : 2018.12.22
#note : add a function
```

## 关键字

```shell
export PATH # 导出变量

#执行完之后，pwd会变
cd ..; ls -all #一行中执行多个命令，使用 ; 隔开

 #使用 ()，相当于父进程fork了一个子进程执行里面的命令，所以此语句执行完之后，pwd没有变
(cd ..; ls -all)
```



## 数据类型

- 只有一种数据类型，string



## 变量

* 全局变量：被 `export` 导出的
* 本地变量：没有被导出的

```shell
VAR=hello; # 定义变量然后赋值
VAR3="hello" #这个和上面那个是一样的
VAR2=$VAR; # $用于取值
VAR4=${VAR2} #取值，加花括号和不加是一样的
unset VAR; # 将定义的环境变量/本地变量删除
```



## 通配符（用于环境名代换）

* `*` 匹配 0 个或任意多个 字符
* `?` 匹配一个任意字符
* `[somechar]` 匹配方括号中任意一个字符的一次出现



## 命令代换

```shell
VAR=`date`; # $VAR 会执行date命令
VAR2=$(date); #完成和上面同一个功能
```



## 算数代换

```shell
VAR=33;
VAR2=$[VAR+40]; #返回的是 代数运算的值
```



## 转义字符

* `\` 
  * 可以续行
  * 可以转义



## 单双引号

* 只是字符串的话，单双引号没啥区别
* 单引和双引可以嵌套
* 单引号：不允许变量扩展
* 双引号：允许变量扩展

```shell
VAR1 = hello;
VAR2 = ;		# 这个代表空

```


## 传参及使用
```shell
$0 # 脚本名
$1 #第一个参数
$# #参数总数，不包含$0
$* #所有参数，不包含$0

```

## 控制流
```shell
for idx in `seq 1 20`
do
  echo "$idx"
done

while [1]
do
  echo "hello world"
done

if [1]
then
 echo "hello world"
else
 echo "hello world if else"
fi


case $VAR in 
yes)
  echo "yes"
  ;;
no)
  echo "no"
  ;;
*)
  echo "input unkown"
  ;;
```

## 条件测试
```shell

if [$VAR1 command $VAR2]

```
**command**
* 对于数值 -eq, -ne, -ge, -le, -gt, -lt
* 对于字符串：=, !=, Z(字符串空), -n(字符串非空)
* 

## 函数
```shell
function new_func()
{
  echo $1 #函数的第一个参数
  echo $2 #函数的第二个参数
  return 0
}

new_func hello world #函数调用
echo $? #获取函数返回值
```
