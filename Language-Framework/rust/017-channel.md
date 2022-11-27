



# std::sync::mpsc

* 无界channel

```rust
fn main() {
    use std::thread;
    use std::sync::mpsc::channel;
    
    // 单生产者(没有调用clone)，单消费者。    
    let (tx, rx) = channel();
    thread::spawn(move || {
        tx.send(10).unwrap();
        tx.send(11).unwrap();
    });
    // 生产者释放之后，会断开channel。所以iter不会阻塞
    for v in rx.iter() {
        println!("{}", v);
    }

    // 多生产者，单消费者
    let (tx, rx) = channel();
    for _ in 0..4 {
        let txc = tx.clone();
        thread::spawn(move || {
            txc.send(10).unwrap();
        });
    }
    // 如果channel没有数据的话，iter会阻塞
    // 实际 rx 套个 Arc<Mutext<..>> 之后也可以出现在多个线程中
    for v in rx.iter() {
        println!("{}", v);
    }

}
```

* 有界channel: channel中的数据右上界，当达到上界，发送端会阻塞。

```rust
fn main() {
    use std::thread;
    use std::sync::mpsc::sync_channel;
    
    let (tx, rx) = sync_channel(2);
    thread::spawn(move || {
        tx.send(10).unwrap();
        tx.send(11).unwrap();
        tx.send(12).unwrap();
        tx.send(13).unwrap();
    });

    for v in rx.iter() {
        println!("{}", v);
    }


    let (tx, rx) = sync_channel(2);
    for _ in 0..4 {
        let txc = tx.clone();
        thread::spawn(move || {
            txc.send(10).unwrap();
        });
    }
    println!("?");
    for v in rx.iter() {
        println!("{}", v);
    }

}
```



# crossbeam::channel

> crossbeam="0.8.2"

```rust
fn main() {
    use crossbeam::channel;
    use std::thread;
    // mpmc，多生产者，多消费者
    // 还有一个 channel::bounded(10).
    let (s, r) = channel::unbounded();
    for _ in 0..4 {
        let s_inner = s.clone();
        thread::spawn(move ||{
            s_inner.send(10).unwrap();
        });
    }

    for i in 0..2 {
        let r_inner = r.clone();
        thread::spawn(move || {
            println!("{}: {}", i, r_inner.recv().unwrap());
        });
    }

    sleep(Duration::new(5, 0));

}
```



# crossbeam::select

> This macro allows you to define a set of channel operations, wait until any one of them becomes ready, and finally execute it. If multiple operations are ready at the same time, a random one among them is selected.



```rust
use crossbeam_channel::{select, unbounded};

let (s1, r1) = unbounded();
let (s2, r2) = unbounded();
s1.send(10).unwrap();

// Block until a send or a receive operation is selected:
// Since both operations are initially ready, a random one will be executed.
select! {
    recv(r1) -> msg => assert_eq!(msg, Ok(10)),
    send(s2, 20) -> res => {
        assert_eq!(res, Ok(()));
        assert_eq!(r2.recv(), Ok(20));
    }
}
```
