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

**更细粒度的时间操作**

```go
package main

import (
	"fmt"
	"strings"
	"time"
)

func main() {
	// Add 时间相加
	now := time.Now()
	// ParseDuration parses a duration string.
	// A duration string is a possibly signed sequence of decimal numbers,
	// each with optional fraction and a unit suffix,
	// such as "300ms", "-1.5h" or "2h45m".
	//  Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".
	// 10分钟前
	m, _ := time.ParseDuration("-1m")
	m1 := now.Add(m)
	fmt.Println(m1)

	// 8个小时前
	h, _ := time.ParseDuration("-1h")
	h1 := now.Add(8 * h)
	fmt.Println(h1)

	// 一天前
	d, _ := time.ParseDuration("-24h")
	d1 := now.Add(d)
	fmt.Println(d1)

	printSplit(50)

	// 10分钟后
	mm, _ := time.ParseDuration("1m")
	mm1 := now.Add(10 * mm)
	fmt.Println(mm1)

	// 8小时后
	hh, _ := time.ParseDuration("1h")
	hh1 := now.Add(hh)
	fmt.Println(hh1)

	// 一天后
	dd, _ := time.ParseDuration("24h")
	dd1 := now.Add(dd)
	fmt.Println(dd1)

	printSplit(50)

	// Sub 计算两个时间差
	subM := now.Sub(m1)
	fmt.Println(subM.Minutes(), "分钟")

	sumH := now.Sub(h1)
	fmt.Println(sumH.Hours(), "小时")

	sumD := now.Sub(d1)
	fmt.Printf("%v 天\n", sumD.Hours()/24)

}

func printSplit(count int) {
	fmt.Println(strings.Repeat("#", count))
}
```



## 时区

* golang 中的时间是有时区的, 在进行 时间之间的计算的时候会考虑到时区的信息

* `"20200513 05:00 +0800"`: 这个串说明: `20200513 05:00` 是东八区的 时间!!!!

```go
time.Now() // 得到一个当前时间, 本地时区
time.Now().In(time.UTC) // 时区转换
time.Now().Location() // 该时间的时区
// 以 time.Local 时区来解释时间 2000010100
ymdh, _ = time.ParseInLocation("2006010215", "2000010100", time.Local)
```



## 时间高级操作



