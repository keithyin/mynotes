# 脚本初步



## 初识

``` shell
#!/usr/bin/env bash
cd ~
mkdir shell_tut
cd shell_tut

for ((i=0; i<10; i++)); do
	touch test_$i.txt
done
```

**可以看出:**

* 命令行中能用的命令在脚本中也是可以用的。





## 变量

* 定义变量

  * `var_name="keith"` 

  * 不需要 `$` 符号

  * 变量名与等号之间不能有空格

* 读取变量
  * `echo $var_name`
  * `echo ${var_name}`  `{}`是可选的，`{}`是为了识别变量的边界 `${var_name}HH`
  * 需要`$` 符号

* 注释

  * `#` 开头

* 字符串

  * `var_name=keith`

  * `var_name='keith'`

  * `var_name="keith"`

  * 可单引，可双引，可不引

  * 单引号中的任何字符都会原样输出，单引号中不能出现单引号

  * 双引号中可以有变量，可以有转移字符
* 拼接字符串
  * `name=$name1$name2`
  * `name="$first$last"`
  * `name="${first}last"`


## 条件控制

