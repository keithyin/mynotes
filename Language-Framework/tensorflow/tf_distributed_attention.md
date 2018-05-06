# 分布式tensorflow注意事项
版本 `tensorflow0.11.0`

适用于 `between-graph`&`synchronous`

(1) 一定要指定 `chief task`

(2) `chief task` 要增加两个`op`:
```python
init_token_op = opt.get_init_tokens_op()
chief_queue_runner = opt.get_chief_queue_runner()
```

(3) `chief task`要执行上面两个`op`:
```python
sv.start_queue_runners(sess, [chief_queue_runner])
sess.run(init_token_op)
```

(4) 使用 `sv.prepare_or_wait_for_session`创建`sess`的时候,一定不要使用`with block`
```python
# wrong
with sv.prepare_or_wait_for_session(server.target) as sess:
  ...
```
会出现错误: 只有`chief task`在训练,`other task`一直打印`start master session...`,不知是什么原因.
```python
# right
sess = sv.prepare_or_wait_for_session(server.target)
```

(5) `opt.minimize()`或`opt.apply_gradients()`的时候一定要传入`global_step`(用来同步的)

(6) 创建`sv`的时候,一定要传入`logdir`(共享文件夹,使用绝对路径).简便方法:传入`log_dir = tempfile.mktemp()`
