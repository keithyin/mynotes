# 神奇的 pytorch



```python
# 叶子 Variable requires_grad=True 的节点不能使用 inplace 操作
a = Variable(torch.randn(5), requires_grad=True)
a[1] = 10 # 对 requires_grad=True叶子节点使用了 inplace操作
b = a+1
torch.sum(b).backward()

```



```python
# 如果不是叶子节点 Variable requires_grad=True, 可以使用 inplace 操作，哪个位置inplace操作了
# 就相当于阻塞那个点之前的路径的梯度传播
a = Variable(torch.randn(3), requires_grad=True)
b = a + 1
b[1] = 10
torch.sum(b).backward()
print(a.grad) # [1, 0, 1] 
```



```python
# 仔细品味下面这个代码
a = Variable(torch.randn(3, 1), requires_grad=True)
b = Variable(torch.randn(1, 3), requires_grad=True)
b = b + 1
b[0] = 10
c = torch.sum(torch.matmul(a, b))
torch.sum(c).backward()
print(a.grad) # 不会报错


a = Variable(torch.randn(3, 1), requires_grad=True)
b = Variable(torch.randn(1, 3), requires_grad=True)
b = b + 1
c = torch.sum(torch.matmul(a, b))
b[0] = 10
torch.sum(c).backward()
print(a.grad) # 会报错

# 仅仅交换了b[0]=10 位置，就会引起错误，这是因为 torch.matmul(a,b) 计算梯度时需要 a,b 当时的值，
# 但是 torch.matmul(a,b) 计算完后 b就被改了，这就搞笑了，导致 torch.matmul(a,b) 的梯度错误的计算，
# 所以会报错
# 但是 第一块就不是这样。
# 主要报错原因是： save_for_backward 的值 在保存后又被修改了。。。。。
```



```python
# 将梯度通过 requires_grad=False 的叶子节点传递出去
a = Variable(init.constant(torch.FloatTensor(5), val=0))
b = Variable(init.xavier_normal(torch.FloatTensor(4, 1)).view(-1), requires_grad=True)
indices = Variable(torch.LongTensor([2, 1]))
a.scatter_(0, indices, b) 
torch.sum(a).backward()
print(b.grad)
# a 的 requires_grad=False, 但是梯度依旧从 a 传到了 b！！！
```

