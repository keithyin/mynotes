# python new 与 init 与 super

```python
demo = Demo()
```

当我们执行上面这条语句的时候，实际上在`python`内部，有两个魔法方法被调用，`__new__` 和 `__init__` 

* `__new__` 用来创建对象实例
* `__init__` 用来初始化对象实例



需要注意的几点：

* `__new__` 需要有返回值， 返回一个实例 (不返回的话，SomeCls() 得到的东西就是 None 了)
* `__init__` 不需要有返回值
* 当为一个类同时写 `__new__` 和 `__init__` 方法时， 注意 `__new__` 也是需要接收参数的
* 如果 `__new__` 返回当前类的实例， 那么 `__init__` 方法对被自动调用
* 如果 `__new__` 返回**不是当前类的实例**， 那么 `__init__` 方法**不会被自动调用 **, 可能需要手动调用一下。





## 参考资料

[http://spyhce.com/blog/understanding-new-and-init](http://spyhce.com/blog/understanding-new-and-init)

[http://howto.lintel.in/python-__new__-magic-method-explained/](http://howto.lintel.in/python-__new__-magic-method-explained/)
[https://www.runoob.com/python/python-func-super.html](https://www.runoob.com/python/python-func-super.html)
