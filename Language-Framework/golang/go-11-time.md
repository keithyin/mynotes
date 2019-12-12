# Time

时间格式：时间格式多种多样，同样 `2018年11月7日` 就可以表示成 `2018/11/07, 2018-11-07` 。

参考时间：`2006年01月02日 03时04分05秒PM  -0700`， 为什么这个是这个参考时间呢？实际上使用 `01,02,03,04,05,06, 07` 分别代表各个字段。

而 Go 语言也是用 参考时间的 `layout` 来表示时间的格式。



**Demo1：将字符串解析成时间（Time对象）, layout 用于表示后面时间的格式**

```go
// func Parse(layout, value string) (Time, error), layout表示 后面时间的格式!!!!
time_obj, err := time.Parse("2006-01-02", "2018-11-07")

// 2018-11-07 19:05:00 +0800 CST, 表示东8区时间
time_obj, err = time.Parse("2006-01-02 03:04PM -0700", "2018-11-07 07:05PM +0800")

// 2018-11-07 19:05:00 +0800 CST
time_obj, err = time.Parse("01-02 03:04PM 06 -0700", "11-07 07:05PM 18 +0800")


// 2018-11-07 00:00:00 +0000 UTC
time_obj, err = time.Parse("2006year 01month 02day ", "2018year 11month 07day")

```



**Demo2：格式化时间，使得打印出来更好看**

```go
// 这是 time_string 中保存的就是：2018year 11month 07day 了
time_string := time_obj.Format("2006year 01month 02day") 
```



```go
// 2018-11-07 11:23:12.4676015 +0800 CST , 当前区域的时间  是 UST+0800 得到
// 所以 UST 时间为 03:23:12
```

**时间计算**
* 今天,明天, 后天, 昨天, 前天, 大前天
```go
// func (t Time) AddDate(years int, months int, days int) Time
currentTime := time.Now()
yestday := currentTime.AddDate(0, 0, -1)  // 数值可正可负, 用来表示日期加还是减
```
