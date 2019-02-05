# python string.format

* `{}` 是占位符， 可以打印， `integer, floating point, string, char, variables`



**语法**

* 语法一：`{}.format(value)`  
* 语法二：`{}{}.format(value1, value2)` 可以打印多个
* 语法三：`{0}{1}.format(value1, value2)`。  `0,1` 为位置参数
* 语法四：`{field_name:conversion}` , `field_name` 即 位置索引，`conversion` 指的是 数据类型的`conversion code` 

> s – strings
> d – decimal integers (base-10)
> f – floating point display
> c – character
> b – binary
> o – octal
> x – hexadecimal with lowercase letters after 9
> X – hexadecimal with uppercase letters after 9
> e – exponent notation
>
> `"This site is {0:f}% securely {1}!!". format(100, "encrypted")`
>
> `"This site is {:f}% securely {}!!". format(100, "encrypted")`



## 参考资料

[https://www.geeksforgeeks.org/python-format-function/](https://www.geeksforgeeks.org/python-format-function/)