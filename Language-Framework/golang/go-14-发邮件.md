# 使用golang发邮件

* `package` : `gomail`
* 安装: `go get github.com/go-gomail/gomail`



```go
package main

import (
	"fmt"

	"github.com/go-gomail/gomail"
)

func SendEmail() {
	message := gomail.NewMessage()
	message.SetAddressHeader("From", "yourmail@qq.com", "AddressHeader的第三个参数")
	message.SetHeader("To",
		message.FormatAddress("othermail@qq.com", "format的是什么?"))
	message.SetHeader("Subject", "什么主题好呢?")
	message.SetBody("text/html", "<b>大佬, 求带</b>")
	dialer := gomail.NewPlainDialer("smtp.126.com", 465, "yourmail@qq.com", "yourpassword")
	if err := dialer.DialAndSend(message); err != nil {
		panic(fmt.Sprintln("sending email error msg=", err))
	}
}
```

https://blog.csdn.net/wangshubo1989/article/details/70808989