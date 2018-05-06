# python String常用方法总结

* `str.split(sep=None, maxsplit=-1)`
将`str`分离,以`sep`作为分离符,返回分离后的列表.
* `str.join(iterable)`
将可迭代对象`iterable`里面的元素了连接成一个`string`.`iterable`中的元素必须是`string`
* str.replace(old, new[, count])
将`str`中的`old`替换成`new`,然后返回一个副本.(不在原`str`上做修改)
* str.strip()

## 参考资料
[https://docs.python.org/3/library/stdtypes.html#string-methods](https://docs.python.org/3/library/stdtypes.html#string-methods)
