# iter,next 与 for

`iter` , `next` 和 `for` 是 `python` 的内置函数



* `iter` : `iter(obj)` 调用 对象的 `__iter__(self)` 方法， 这个方法一般返回一个 可迭代对象。
* `next`：`next(obj)` 调用对象的  `__next__(self)` 方法，一般有这个方法的 对象都可叫做 可迭代对象。



关于 for ，其实可以这么看：

```python
for i in obj:
	...

# 等价于
iterable_obj = iter(obj)
while True:
    try:
        i = next(iterable_obj)
    except:
        break
    # do something
```

