# 异常处理：Exception

* Error：严重错误，JVM内部错误，资源耗尽
* Exception：空指针访问，试图读取不存在的文件，网络连接中断



* Exception 和 Error 都继承了 Throwable
  * Error：错误，程序中不作处理
  * Exception：异常，程序中捕获，处理
    * 编译异常
    * 运行时异常



**java异常处理模型：抓抛模型**

* 抛：执行代码时，一旦出现异常，就会在异常的代码位置生成一个对应异常的对象，并将此对象抛出。
  * 一旦抛出，下面的代码就不执行了。
  * 抛给 函数的调用者

```java
// throws 后面是异常类型列表，既然 throw，里面就不需要 try catch 了
void func throws FileNotFindException, IOException{
    // 这里有可能会抛异常的代码
}

// 手动抛，抛对象，这时候不需要 异常列表
void func2(){
    throw new RuntimeException("exception");
}
```





* 抓：抓抛出来的异常对象，然后处理，两种处理方式
  * 自己处理异常
  * 继续往外抛



* 多个 `catch` 只会进去一个

```java
try{
    // 可能出现错误的代码
}catch(Exception e){
    // 处理异常
    e.printStackTrace()
}catch(Exception e){
    
}finally{
  //一定会执行的代码，无论抛不抛异常   
}
```

