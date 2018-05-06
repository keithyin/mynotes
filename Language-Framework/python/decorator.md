# python 装饰器
学识浅薄,如有错误,请不吝指正,谢谢!

**什么是装饰器:**
`python`中的装饰器可以理解成函数的一个`wrapper`.

**如何写一个装饰器:**
知道了`python`中的装饰器是函数的一个`wrapper`之后,那么在写装饰器的时候,当然是需要将函数当作参数的.
```python
def log(func):
    def wrapper(*args, **kw):
        print('do something before call %s ():' % func.__name__)
        return func(*args, **kw)
    return wrapper
```
这是一个简单的装饰器,仔细观看这段代码的特点
输入: 函数对象
返回: 一个函数对象(原函数对象的`wrapper`).

**如何将这个装饰器用到函数上**
```python
@log
def say():
  print('hello')
```
将`@log`放到`say()`函数的定义之前,就等价于执行了`say = log(say)`



**如果 decorator 需要传入参数**

```python
def log(text):
    def decorator(func):
        def wrapper(*args, **kw):
            print('%s %s():' % (text, func.__name__))
            return func(*args, **kw)
        return wrapper
    return decorator

@log("do something") #会调用 log(text) 返回 decorator, 然后 decorator 才作用到 func 上.
def func():
    pass
"""
相当于执行了: func = log("do something")(func)
"""
```