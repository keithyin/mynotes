### 内置锁

> java提供了内置的锁机制来支持原子性，同步代码块。
>
> 同步代码块包含两个部分：1. 作为锁的对象引用，2.一个由这个锁保护的代码块。
>
> 以 `synchronized` 来修饰的方法就是一种横跨整个方法体的同步代码块，其中该同步代码块的锁就是方法调用所在的对象。
>
> 静态的 `synchronized` 方法以 `Class` 对象作为锁



```java
synchronized(lock) {
  
}
```



内置锁是可重入的，如果某个线程试图获取一个他已经持有的锁，那么请求会成功！

> Pthread上的互斥锁是不可重入的。
>
> 可重入的锁：是线程粒度的锁
>
> 不可重入的锁：是调用粒度的锁

```java
public class Widget {
  public synchronized void doSomething() {
    
  }
}


public class LoggingWidget extends Widget {
  public synchronized void doSomething() {
    System.out.println(toString() + ": calling doSomething");
    super.doSomething() // 如果不可重入，这里会阻塞
  }
}

```



* 加锁机制可以保证可见性 & 原子性。`volatile` 只能保证可见性



> 发布与溢出
>
> * 发布一个对象：使对象能够在 **其创建的作用域** 以外的作用域中使用
> * 溢出：不该发布的对象被发布
>   * 什么叫不该被发布？



```java
// 发布一个对象
// 将其保存到一个公有的静态变量中

public static Set<Secret> knownSecrets;

public void initialize() {
  knownSecrets = new HashSet<Secret>();
}
```



```java
// 溢出
// 既然 states 已经声明了 private，那肯定是不希望 外面直接访问的。但是 getStates 却将其引用返回出去了
class UnsafeStates {
  private String[] states = new String[]{'hello', 'world'}
  
  public String[] getStates() {return states;}
}
```



```java
// This 溢出，没有看懂为啥就溢出了。。。。。
public class ThisEscape {
  public ThisEscape(EventSource source) {
    source.registerListener() {
      new EventListener() {
        public void onEvent(Event e) {
          doSomething(e);
        }
      }
    }
  }
}
```



> 线程封闭
>
> 当访问共享的可变数据时，通常需要同步。一种避免使用同步的方式就是不共享数据。如果仅在单线程中访问数据，也不需要同步。这种技术被称之为 线程封闭(Thread Confinement)



```java
// ThreadLocal 维持线程封闭性。比较适合存放上下文信息。常用作全局变量
// 和 c、c++ 中的 thread_local 关键字一致
private static ThreadLocal<Connection> connectionHolder = 
  new ThreadLocal<Connection>() {
  	public Connection initialValue(){
      return DriverManager.getConnection(DB_URL);
    }
	};

public static Connection getConnection() {
  return connectionHolder.get();
}
```



### 服务器

```java
class SingleThreadWebServer {
  public static void main(String[] args) throws IOException {
    ServerScoket socket = new ServerSocket(80);
    while (true) {
      Socket conn = socket.accept();
      handleRequest(conn);
    }
  }
}
```



```java
// 为每个任务创建一个线程。
// 当创建大量线程时性能会下降
class ThreadPerTaskWebServer {
  public static void main(String[] args) throws IOException {
    ServerScoket socket = new ServerSocket(80);
    while (true) {
      final Socket conn = socket.accept();
      Runnable task = new Runnable() {
        public void run() {
          handleRequest(conn);
        }
      }
      new Thread(task).start()
    }
  }
}
```



#### Executor

> 将任务提交和任务执行解耦。



```java
public interface Executor {
  void execute(Runnable command); // 这里负责实现各种 执行策略！
}
```





```java
class TaskExecutionWebServer {
  private static final int NTHREADS = 100;
  private static final Executor exec = Executors.newFixedThreadPool(NTHEADS);
  
  public static void mian(String[] args) throws IOException {
    ServerScoket socket = new ServerSocket(80);
    while (true) {
      final Socket conn = socket.accept();
      Runnable task = new Runnable() {
        public void run() {
          handleRequest(conn);
        }
      }
      exec.execute(task);
    }
  }
}
```



> Executors 中的工厂方法
>
> * `newFixedThreadPool`: 固定长度的线程池
> * `newCachedThreadPool`: 会自动回收空闲线程 & 添加新的线程
> * `newSingleThreadExectuor`: 单线程的 `Exectuor`
> * `newScheduledThreadPool`: 固定长度的线程池，以延迟或者定时的方式来执行任务



* Executor 如何关闭： `ExecutorService`

