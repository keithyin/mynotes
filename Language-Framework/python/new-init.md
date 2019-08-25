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

```python
class A:
    _instace = None

    def __new__(cls, a, b):
        print("A.__new__")
        if not cls._instace:
            # 创建实例, __new__(cls) ，这个 cls 也可以是其它的类，那就就实例化出来的一个其它的类别的对象
            cls._instace = super(A, cls).__new__(cls)
        print("A.__new__ out")
        # 返回之后会嗲用init初始化实例
        # 如果返回的不是此类的实例，则不会调用init
        return cls._instace

    def __init__(self, a, b):
        print("A.__init__ enter")
        self.a = a
        self.b = b
        print("A.__init__ out")

    def __str__(self):
        return ("a:{}, b:{}".format(self.a, self.b))


a = A(1, 2)
print(a)
```



## 参考资料

[http://spyhce.com/blog/understanding-new-and-init](http://spyhce.com/blog/understanding-new-and-init)

[http://howto.lintel.in/python-__new__-magic-method-explained/](http://howto.lintel.in/python-__new__-magic-method-explained/)

[https://www.runoob.com/python/python-func-super.html](https://www.runoob.com/python/python-func-super.html)
