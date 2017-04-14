# tensorflow分布式中几个重要函数总结
### tf.train.Server.create_local_server()
```python
tf.train.Server.create_local_server(config=None, start=True)
#config:A tf.ConfigProto that specifies default configuration options for all sessions that run on this server.
# config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
#log_device_placement：是否打印变量的放置信息
#allow_soft_placement:
```
