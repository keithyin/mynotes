# 远程过程调用

* 客户端执行一段代码
* 底层会将指令通过网络传给服务器
* 服务器执行操作，再将结果传给客户端
* 实际上是在服务器上执行的操作，就像在本地执行的一样



* RPC 就在 hadoop 的 common.jar 包里

```java
// 服务器端
// LoginServiceInterface.java
// 留着写业务代码的接口
public interface LoginServiceInterface{
    // 定义协议的版本号
    public static final long versionID=1L;
    public String login(String username, String password);
}

// LoginServiceImpl.java
public class LoginServiceImpl implements LoginServiceInterface{
    @override
    public String login(String username, String password){
        return username + "logged in successfully."
    }
}

// Starter.java, 将类发布为服务
public class Starter{
    public static void main(String[] args){
        Builder builder = new RPC.Builder(new Configuration);
        builder.setBindAddress("192.168.204.200"); //绑定地址
        builder.setPort(10000); //服务端口
        builder.setProtocol(LoginServiceInterface.class);// 协议，就是业务接口
        builder.setInstance(new LoginServiceImpl); // 具体的业务实例
        Server server = builder.build();
        server.start(); // 启动一个 socket 程序
    }
}
```



```java
// 客户端
// LoginServiceInterface.java
// 留着写业务代码的接口
public interface LoginServiceInterface{
    // 定义协议的版本号
    public static final long versionID=1L;
    public String login(String username, String password);
}

// LoginController.java
public class LoginController{
    public static void main(String[] args){
        // 1L 意思是 client version
        LoginServiceInterface proxy = RPC.getproxy(LoginServiceInterface.class, 1L,
                     new InetSocketAddress("192.168.204.200", 10000),
                    new Configuration());
     	String result = proxy.login("keith", "123456");
        
    }
}
```

