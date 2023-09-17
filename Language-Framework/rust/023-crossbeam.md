```toml
[dependencies]
crossbeam = "0.8"
```

# MPMC (multi-producer multi-consumer channels)
https://docs.rs/crossbeam/latest/crossbeam/channel/index.html 

```rust

use crossbeam_channel::unbounded;

// Create a channel of unbounded capacity.
let (s, r) = unbounded();

// Send a message into the channel.
s.send("Hello, world!").unwrap();

// Receive the message from the channel. block until the message arrived
assert_eq!(r.recv(), Ok("Hello, world!"));

```

## 共享
```rust
// 共享，通过 .clone() 进行共享

use std::thread;
use crossbeam_channel::bounded;

let (s1, r1) = bounded(0);
let (s2, r2) = (s1.clone(), r1.clone());

// Spawn a thread that receives a message and then sends one.
thread::spawn(move || {
    r2.recv().unwrap();
    s2.send(2).unwrap();
});

// Send a message and then receive one.
s1.send(1).unwrap();
r1.recv().unwrap();

// 也可以通过 引用进行共享
use crossbeam_channel::bounded;
use crossbeam_utils::thread::scope;

let (s, r) = bounded(0);

scope(|scope| {
    // Spawn a thread that receives a message and then sends one.
    scope.spawn(|_| {
        r.recv().unwrap();
        s.send(2).unwrap();
    });

    // Send a message and then receive one.
    s.send(1).unwrap();
    r.recv().unwrap();
}).unwrap();

```
## 关闭channel
```rust
// 关闭channel，不能再发送了，但是已经发送的可以被读取。全部读完的话，recv 操作会返回Err
// 在 共享 sender 的场景下是什么样的？难道是所有的sender都关闭之后，才能不发消息？

use crossbeam_channel::{unbounded, RecvError};

let (s, r) = unbounded();
s.send(1).unwrap();
s.send(2).unwrap();
s.send(3).unwrap();

// The only sender is dropped, disconnecting the channel.
drop(s);

// The remaining messages can be received.
assert_eq!(r.recv(), Ok(1));
assert_eq!(r.recv(), Ok(2));
assert_eq!(r.recv(), Ok(3));

// There are no more messages in the channel.
assert!(r.is_empty());

// Note that calling `r.recv()` does not block.
// Instead, `Err(RecvError)` is returned immediately.
assert_eq!(r.recv(), Err(RecvError));

```

## blocking操作

send 和 recv 的操作都有三类：
1. Non-Blocking （return immediately with success or failure）
2. Blocking (waits until the operation succeeds or the channel becomes disconnected)
3. blocking with a timeout (blocks only for a certain duration of time)

```rust
use crossbeam_channel::{bounded, RecvError, TryRecvError};

let (s, r) = bounded(1);

// Send a message into the channel.
s.send("foo").unwrap();

// This call would block because the channel is full.
// s.send("bar").unwrap();

// Receive the message.
assert_eq!(r.recv(), Ok("foo"));

// This call would block because the channel is empty.
// r.recv();

// Try receiving a message without blocking.
assert_eq!(r.try_recv(), Err(TryRecvError::Empty));

// Disconnect the channel.
drop(s);

// This call doesn't block because the channel is now disconnected.
assert_eq!(r.recv(), Err(RecvError));
```

## 迭代读取

```rust
use std::thread;
use crossbeam_channel::unbounded;

let (s, r) = unbounded();

thread::spawn(move || {
    s.send(1).unwrap();
    s.send(2).unwrap();
    s.send(3).unwrap();
    drop(s); // Disconnect the channel.
});

// Collect all messages from the channel.
// Note that the call to `collect` blocks until the sender is dropped.
let v: Vec<_> = r.iter().collect();

assert_eq!(v, [1, 2, 3]);
```

## Selection

## Extra channels

# 

