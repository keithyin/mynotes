# matlab variable

## The format Command

默认情况下，`matlab`仅显示到小数点后四位，这就是 `format short`.

**注意，format语法仅仅控制数据的显示，不代表数据的存储**

```matlab
>>format short
>>x = 1.23423424;
>>x
1.2342

>>format long %显示到小数点后16位
>>x
1.234234240000000

>>format short e %科学显示
>>x
1.2342e+00

>> format long e
```



## 如何创建向量 vector

* 使用中括号

```matlab
a = [1 2 3 4]
b = [1, 2, 3, 4] %和上面那个是等价的

c = [1; 2; 3; 4] %创建的是列向量

```



## 如何创建矩阵

* 矩阵就是二维的向量

```matlab
a = [1 2; 3 4]
b = [1,2; 3,4]
```



## 参考资料

[https://www.tutorialspoint.com/matlab/matlab_variables.htm](https://www.tutorialspoint.com/matlab/matlab_variables.htm)

