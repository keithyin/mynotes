# SSH 登录原理

验证方法：

* 密码验证(client 输入用户名密码登录)
* 秘钥身份验证机制 （client保存一个秘钥，作为合法授权凭证）



**密码验证**

```shell
ssh user@ip# 本地执行
# 然后输入密码
```



**秘钥登录**

```shell
# 1. client 生成一对秘钥，（公钥和私钥）
# 2. 把公钥放在 remote 授权列表里面 (authorized_keys)
# 之后直接 ssh 就可以登录了，不需要密码了。
```

```shell
# 在 client 上
ssh-keygen -t rsa #生成ssh密钥对，rsa 是加密算法
# 使用 scp 将 .pub 发送到 remote 中

# 在 remote 上
cd .ssh
cat *.pub >> authorized_keys # 追加一个 pub
# authorized_keys 的权限是 600
```

