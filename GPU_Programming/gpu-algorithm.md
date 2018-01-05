# 基本 GPU 算法

* reduce
* scan



## reduce

* inputs :  **set of elements** , **reduction operators**
* outputs : **one elements**

```python
a = [1,2,3,4,5]
op = add
res = 15
```





## scan

* inputs :   **set of elements** , **scan operators**， **identity element**
  * I op element = element
* outputs : **set of elements** 

```python
a = [1,2,3,4,5]
op = add
res = [1,3,6,10,15] # inclusive
# or
res = [0,1,3,6,10] # exclusive
```

**特点：**

* 下一个输出 和 上一个输出 有关

|                 | More step efficient | more work efficient |
| --------------- | ------------------- | ------------------- |
| Hillis + Steele | no                  | Yes                 |
| Blelldch        | Yes                 | no                  |









