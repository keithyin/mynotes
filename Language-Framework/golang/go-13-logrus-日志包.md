# logrus日志包

**安装**

```
# clone到 GOROOT/src/github.com 或者 GOPATH/src/github.com 下
git clone git@github.com:sirupsen/logrus.git

# 然后在 go代码中就可以 import "github.com/logrus" 然后 logrus.    使用了
```



```go
package main

import (
  "os"
  log "github.com/sirupsen/logrus"
)

func init() {
  // Log as JSON instead of the default ASCII formatter.
  log.SetFormatter(&log.JSONFormatter{})

  // Output to stdout instead of the default stderr
  // Can be any io.Writer, see below for File example
  log.SetOutput(os.Stdout)

  // Only log the warning severity or above.
  log.SetLevel(log.WarnLevel)
}

func main() {
  log.WithFields(log.Fields{
    "animal": "walrus",
    "size":   10,
  }).Info("A group of walrus emerges from the ocean")

  log.WithFields(log.Fields{
    "omg":    true,
    "number": 122,
  }).Warn("The group's number increased tremendously!")

  log.WithFields(log.Fields{
    "omg":    true,
    "number": 100,
  }).Fatal("The ice breaks!")

  // A common pattern is to re-use fields between logging statements by re-using
  // the logrus.Entry returned from WithFields()
  contextLogger := log.WithFields(log.Fields{
    "common": "this is a common field",
    "other": "I also should be logged always",
  })

  contextLogger.Info("I'll be logged with common and other field")
  contextLogger.Info("Me too")
}
```





## Fields

Logrus encourages careful, structured logging through logging fields instead of long, unparseable error messages. For example, instead of: `log.Fatalf("Failed to send event %s to topic %s with key %d")`, you should log the much more discoverable:

```go
log.WithFields(log.Fields{
  "event": event,
  "topic": topic,
  "key": key,
}).Fatal("Failed to send event")
// 打印出来的结果为（结构化，看起来比较舒服）
// time="2019-07-24T23:00:02+08:00" level=fatal msg="failed to send" event=open topic=learn key=sonething
```



## 日志级别

```go
log.Trace("Something very low level.")
log.Debug("Useful debugging information.")
log.Info("Something noteworthy happened!")
log.Warn("You should probably take a look at this.")
log.Error("Something failed but I'm not quitting.")

// Calls os.Exit(1) after logging
log.Fatal("Bye.")

// Calls panic() after logging
log.Panic("I'm bailing.")
```

**设置日志级别**

```go
//设置日志级别
log.SetLevel(log.InfoLevel)
```



## Formatters

* logrus.TextFormatter.文本格式记录日志（输出tty时会加颜色）
* logrus.JSONFormatter. 以json格式记录日志



## 输出

```go
// log.Setoutput(filehandle)
log.SetOutput(os.Stdout)
```



## 使用

```go
// 可以引包直接使用
logrus.Fatal(".......")

//也可以
log:=logrus.New()
log.Fatal("......")
```

