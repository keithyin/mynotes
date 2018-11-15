# HTTP协议

* 建立在 `TCP` 协议基础上的

**Demo1：浏览器发送了几次http请求**

```html
# demo.html
<h1>abc</h1>
<img src="sun.jpg"/>
<img src="sum2.jpg"/>
```

* 当使用浏览器访问在服务器上的这么一个页面时，会发送三次请求
  * 第一次请求文本
  * 然后发现有图片，会发送两次请求去获取图片内容



## Http请求

包含三个部分

* 一个请求行
* 若干求请求头
* 实体内容（要传给服务器的数据）

**Demo2: 一个 Http请求头 **

```http
GET / HTTP/1.1
Host: www.uestc.edu.cn
Connection: keep-alive
Upgrade-Insecure-Requests: 1
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36
Accept:  text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8
Accept-Encoding: gzip, deflate, br
Accept-Language: en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7
Cookie: UM_distinctid=16508b421811aa-0c9b597f30b2ac-47e1039-e1000-16508b421828e; CNZZDATA5947016=cnzz_eid%3D767141219-1520423712-null%26ntime%3D1537677946; __utma=108824541.2143841325.1493785358.1541074139.1541132894.47; __utmz=108824541.1541132894.47.24.utmcsr=idas.uestc.edu.cn|utmccn=(referral)|utmcmd=referral|utmcct=/authserver/login

如果是 POST 请求的话，数据会呈现在这里
```

* Accept： 表示客户端可以接收哪些数据类型
* Referer：表明是从哪个网址来的，假设在 `www.baidu.com`点击的超链接，就会变成这样      `referer:www.baidu.com` ， （根据实际情况决定有没有）
* Accept-Language：接收什么语言
* Accept-Encoding：接收的压缩方式
* Host：要找的主机
* Connection：keep-alive，不要立即断开请求



HTTP请求，GET与POST

* Get请求数据会显示在地址栏上，Post的请求数据是在 请求包的 实体内容里面
* 数据量比较大的话使用 POST 请求！！
* **在使用 AJAX 的时候，如果使用 GET 方式发请求，如果 URL 的值不变，浏览器不会真的去发请求，而是会从缓存中取数据**
  * 解决方法是在 URL 上添加一个时刻在变的值，比如，随机数

## HTTP响应

**Demo3：HTTP响应头**

```http
HTTP/1.1 200 OK
Server: nginx
Date: Sun, 04 Nov 2018 08:46:44 GMT
Content-Type: text/html; charset=utf-8
Last-Modified: Sun, 04 Nov 2018 04:19:43 GMT
Transfer-Encoding: chunked
Connection: keep-alive
Vary: Accept-Encoding
ETag: W/"5bde735f-67ef"
Content-Encoding: gzip

消息内容。。。
```

* 状态码说明
  * 302：资源被转移到了新地址
  * 404：页面不存在
  * 304：如果本地有缓存，而且是最新，就不用从服务器取数据了
* Location：302状态码与 Location字段一起使用
  * 客户端看到 302，就可以再发送HTTP请求，访问 Location 的地址
* Last-Modified：请求的资源最近更新时间





## Cookie

在 **会话** 过程中会产生 一些数据。

Cookie：服务器将数据写给浏览器。

* 如果服务器希望客户端保存一些数据
* 就可以将数据以 cookie 的形式返回给 浏览器
* 浏览器看到接收的是 cookie，就会保存在本机上。
* cookie 保存在本机上就是字符串



如果在一次访问此网站

* 浏览器就会将保存在本机上的 cookie 包在 HTTP请求 中发送过去



**Demo4：cookie，PHP代码**

```php
// 服务器
<?php
    // 保存 cookie
    // key:value , 第三个参数表示在客户端保存时间，按秒计算
	setCookie("name", "keith", time()+3600)
?>
```

* 如果再一次访问之前的网站，浏览器会将此网站之前保存的 cookie 封装到 HTTP 请求头上，发给服务器。

```php
<?php
 $val = $_COOKIE['key']    // 获取指定的键对应的值
?>
```



# JavaScript 运行原理

* 实现网页动态效果的基石，常用于给网页添加动态功能
* 监听鼠标的响应，键盘的响应
* 用于客户端 WEB 开发的脚本语言，JavaScript 的解释器引擎来自浏览器
  * JavaScript 基本都是在客户端浏览器执行的
  * JavaScript 是嵌入在 HTML 页面中的
  * 在客户端请求时，服务器会将 HTML 页面（里面包含JS代码）返回，然后浏览器会执行返回的HTML页面。这时候，浏览器也会去解析JS代码了。

**DOM: Document Object Model**

* 会将文档看做一个 DOM 树，通过对 DOM 对象的操作，就会操作文档
* 用户可以对 HTML 元素进行控制：变大，变小，添加，删除
* `Document` 对象用来代表整个 `HTML` 文档，使用 `Document` 对象可以操作 `HTML` 中的任何元素
* [https://www.w3schools.com/js/js_htmldom_document.asp](https://www.w3schools.com/js/js_htmldom_document.asp)



[**JQuery**](http://www.runoob.com/jquery/jquery-syntax.html）

```javascript
// JQuery，同时满足两个 class 的选择器
$(".class1.class2")

```



**goquery**

```go
/*
goquery.Selection: Selection represents a collection of nodes matching some criteria
html.Node：表示 html 中的一个节点
*/
```







# AJAX 运行原理

* 异步的 JavaScript 和 XML

**七个技术的综合**

* JavaScript，XML，XSTL，XHTML, DOM, XMLHTTPREQUEST, CSS

**解决的问题**

* 原始方法的 **全局刷新** 问题
* 原始方法，数据同时提交到服务器
* 可以进行，**无刷新的数据交换技术**



**AJAX 可以给客户端返回三种格式的数据**

* 文本
* XML
* JSON



**AJAX**

* 传统方法的缺点，一个 请求，对应一个响应（响应要返回整个页面）
* AJAX 基本流程
  * 创建 AJAX 引擎对象（XMLHttpRequest对象）
  * 将数据发送到 服务器（对应的 php文件）
  * HTTP响应（由服务器代码决定）
  * 回调函数处理返回的结果
* 请求 --> AJAX引擎对象 ---http请求--> 服务器 ---http响应---> AJAX引擎对象 --> 放到指定的位置
  * 响应不需要刷新整个页面，只需要返回关键数据即可，然后使用DOM编程更新页面的数据
  * HTTP请求实际是：XMLHTTPRequest, 本质还是 HTTP 请求
  * 这里的 HTTP 响应有三种格式：文本，XML，JSON

```javascript
// js 代码，由客户端执行，
// 因为 XMLHttpRequest 实际上也是 HTTP请求，所以服务器的处理方式和 HTTP 请求没有什么不同
function ajax(){
    var xmlHttp = new XMLHttpRequest()
    // 第一个参数：请求方式 get/post。第二个参数：url。第三个参数：是否使用异步
    xmlHttp.open("get", "/ajax/test.php?username=something", true)

    // 设置回调函数，指定返回的结果应该如何处理
    xmlHttp.onreadystatechange=recallfunction

    // 发送请求, 如果 get 请求，null即可， 如果 post 请求，填入实际数据
    xmlHttp.send(null)
}

function recallfunctioin(){
    
}
```





