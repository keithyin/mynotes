# linux 终端编码问题

**问题： locale cannot set LC_ALL to default locale no such file or directory debian**

```shell
locale-gen en_US.UTF-8
```

**如果找不到 locale-gen command， 先执行，然后再执行上面那句**

```shell
apt-get clean && apt-get update && apt-get install -y locales
```





[https://perlgeek.de/en/article/set-up-a-clean-utf8-environment](https://perlgeek.de/en/article/set-up-a-clean-utf8-environment)

[https://unix.stackexchange.com/questions/110757/locale-not-found-setting-locale-failed-what-should-i-do](https://unix.stackexchange.com/questions/110757/locale-not-found-setting-locale-failed-what-should-i-do)