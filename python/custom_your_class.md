# Python class 总结

## 细数class中的 `__**__`

1. `__init__(self, *values)`
对象的初始化函数,初始化类的实例时,会调用这个方法
2. `__str__(self)`
返回 print对象时要打印的东西,pirnt(obj)时会调用这个方法
3. `__iter__(self)` 与 `__next__(self)`
将对象 变为可迭代对象,`__iter__()`用于`iter()`,`__next__`用于`next()`
4. `__getitem__(self, key)`
使得对象可以向类表一样使用,obj[2], obj[2:4]
5. `__setitem__(self, key, value)`
可以使对象的key被赋值,obj['hello'] = 'world'
6. `__getattr__(self, attr)`
如果对象没有所调用的属性的时候,就会把属性名送进这个方法,看看这个方法返回什么
7. `__setattr__(self, name, value)`
当给一个属性赋值的时候,如果类的实例中没有这个属性时,就会调用这个方法.
8. `__call__(self, *value)`
使得类的实例可以像函数一样被调用
9. `__slots__`
这个和前几个不太一样,它是个类的属性,不是方法,这个用来限制类的实例所能动态添加的属性

## 细数用在class中的装饰器
1. `@property`
是一个装饰器,将一个方法当作属性调用
2. `@staticmethod`
将方法定义成静态方法,不需要传入`self`了
3. `@classmethod`
将一个方法定义成类方法,传入`cls`而不是`self`
4. `@abstraction`
将一个方法声明成抽象方法

## 参考资料
[http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/00143186739713011a09b63dcbd42cc87f907a778b3ac73000](http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/00143186739713011a09b63dcbd42cc87f907a778b3ac73000)
[http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/00143186781871161bc8d6497004764b398401a401d4cce000](http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/00143186781871161bc8d6497004764b398401a401d4cce000)
[http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014319098638265527beb24f7840aa97de564ccc7f20f6000](http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014319098638265527beb24f7840aa97de564ccc7f20f6000)
[https://jeffknupp.com/blog/2014/06/18/improve-your-python-python-classes-and-object-oriented-programming/](https://jeffknupp.com/blog/2014/06/18/improve-your-python-python-classes-and-object-oriented-programming/)
[http://stackoverflow.com/questions/100003/what-is-a-metaclass-in-python](http://stackoverflow.com/questions/100003/what-is-a-metaclass-in-python)
[https://docs.python.org/3/library/functions.html#setattr](https://docs.python.org/3/library/functions.html#setattr)
