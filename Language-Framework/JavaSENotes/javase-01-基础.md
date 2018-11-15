# javase 基础

## 数据类型

* 基本数据类型 vs 引用数据类型（数组，类，接口）
* 成员变量  vs 局部变量

**基本数据类型**

* byte, short, int, long
* float, double
* char , `''`
* boolean



**默认值**

* byte short int long ==>0

 *            float double ==>0.0
 *            char ==>空格
 *            boolean ==>false 
 *            引用类型变量==>null



## 数组

```java
int[] i = new int[12]; //12个元素的数组
int[] j = new int[]{12, 3}; //2个元素的数组

String[][] str = new String[4][3]; //四行三列

// 为啥不用指定列就可以，因为在二维数组中，保存每行的首地址就足够了
// 只指定行说明只是分配了 保存地址的 内存空间
String[][] str2 = new String[4][]; // 4行，不用指定列
str1[0] = new String[3];
str1[1] = new String[5]; //可以不一致哦！！！


int[][] arr = new int[][]{{1,2,3}, {4,5}, {6}}; // 三行三列
```



## 流程控制

* `if else`
* `switch case`

```java
if (expression){
    
}else if (expression2){
    
}else{
    
}

// 变量可以是以下类型：byte short int char enum String
switch (variable){
    case val1:
        break;
    case val2:
        break;
    default:
        break
}
```

* `for`
* `while`
* `do ... while`:  和 `while` 相比，至少会执行一次

```java

// 初始化，判断条件，执行循环体，step+1，判断条件，执行循环体，step+1。。。
for (;;){
}

while (){
    
}

do{
    
}while()
```





## 内存布局

* 栈
* 堆
* 方法区
  * 字符串常量池
* 静态域：放静态变量的