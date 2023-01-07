# 标准库对于async的支持

1. `trait Future, enum Poll, struct Context, struct Waker`

2. `async/.await`

```rust
pub trait Future {
    /// The type of value produced on completion.
    type Output;
    /// Attempt to resolve the future to a final value, 
    /// registering the current task for wakeup if the value is not yet available.
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}

pub enum Poll<T> {
    /// Represents that a value is immediately ready.
    Ready(#[stable(feature = "futures_api", since = "1.36.0")] T),

    /// Represents that a value is not ready yet.
    ///
    /// When a function returns `Pending`, the function *must* also
    /// ensure that the current task is scheduled to be awoken when
    /// progress can be made.
    Pending,
}

/// 异步task的上下文
///
/// 当前，`Context` 只负责提供对于 &Waker 的访问
/// &Waker 是用来唤醒当前 task
#[stable(feature = "futures_api", since = "1.36.0")]
pub struct Context<'a> {
    waker: &'a Waker,
    // Ensure we future-proof against variance changes by forcing
    // the lifetime to be invariant (argument-position lifetimes
    // are contravariant while return-position lifetimes are
    // covariant).
    _marker: PhantomData<fn(&'a ()) -> &'a ()>,
}

// A Waker is a **handle** for waking up a task 
// Waker是一个 handle(句柄)
// 目的是：唤醒一个task
// 通过什么做到：通过通知 executor 该 task 可以继续执行
// Waker里面的 wake, wake_by_ref 都是调用 RawWaker里面对应的方法来实现的
pub struct Waker {
    waker: RawWaker,
}

// RawWaker 允许 implementor of a task executor 去创建一个 Waker
// 新创建的 Waker 提供定制化的 wakeup 行为
// 一个原始数据的指针！，一个虚表
pub struct RawWaker {
    /// A data pointer, which can be used to store arbitrary data as required
    /// by the executor. This could be e.g. a type-erased pointer to an `Arc`
    /// that is associated with the task.
    /// The value of this field gets passed to all functions that are part of
    /// the vtable as the first parameter.
    data: *const (), // 这个指针指向的是Task
    /// Virtual function pointer table that customizes the behavior of this waker.
    vtable: &'static RawWakerVTable, // 这个table存储的也是Task里面的对应方法
}

// 虚表，一个结构体，里面都是函数指针。好奇的一点是这里为啥不用 trait。
pub struct RawWakerVTable {
    /// This function will be called when the [`RawWaker`] gets cloned, e.g. when
    /// the [`Waker`] in which the [`RawWaker`] is stored gets cloned.
    ///
    /// The implementation of this function must retain all resources that are
    /// required for this additional instance of a [`RawWaker`] and associated
    /// task. Calling `wake` on the resulting [`RawWaker`] should result in a wakeup
    /// of the same task that would have been awoken by the original [`RawWaker`].
    clone: unsafe fn(*const ()) -> RawWaker,

    /// This function will be called when `wake` is called on the [`Waker`].
    /// It must wake up the task associated with this [`RawWaker`].
    ///
    /// The implementation of this function must make sure to release any
    /// resources that are associated with this instance of a [`RawWaker`] and
    /// associated task.
    wake: unsafe fn(*const ()),

    /// This function will be called when `wake_by_ref` is called on the [`Waker`].
    /// It must wake up the task associated with this [`RawWaker`].
    ///
    /// This function is similar to `wake`, but must not consume the provided data
    /// pointer.
    wake_by_ref: unsafe fn(*const ()),

    /// This function gets called when a [`RawWaker`] gets dropped.
    ///
    /// The implementation of this function must make sure to release any
    /// resources that are associated with this instance of a [`RawWaker`] and
    /// associated task.
    drop: unsafe fn(*const ()),
}


```



整体概念：

1. Executor 应该有一个任务队列，不停的 for loop 来遍历该队列，取出任务来执行
   
   1. 取出Task的时候，得到任务相关的waker。
   
   2. 从任务中取出 Future，poll 一下，如果 pending。将其扔到一个独立的线程，就不管了， 该线程来检查Future是不是可以继续走了，如果可以继续走了，就将其唤醒（Waker负责将该Task重新扔回Executor的执行队列）。

2. Task 表示任务，里面得包含一个 Future, Executor来调度Task。Waker也得在Task里。
   
   1. 或者Future自己是Task，Future自己包含一个 Waker
   
   2. Task 要实现 ArcWake. `impl ArcWake for Task`.  Waker的核心就是将自己发送到Executor 的队列中！



# RawWakerVTable

虚表：虚表在C++中是一个函数指针表，指向虚函数，RawWakerVTable也是类似作用

```rust
/// 虚表：里面的成员都是函数指针，这也符合虚表的定义，存了一堆函数指针
pub struct RawWakerVTable {    
    clone: unsafe fn(*const ()) -> RawWaker,
    wake: unsafe fn(*const ()),
    wake_by_ref: unsafe fn(*const ()),
    drop: unsafe fn(*const ()),
}
```

关于 RawWaker

```rust
pub struct RawWaker {
    /// A data pointer, which can be used to store arbitrary data as required
    /// by the executor. This could be e.g. a type-erased pointer to an `Arc`
    /// that is associated with the task.
    /// The value of this field gets passed to all functions that are part of
    /// the vtable as the first parameter.
    data: *const (),
    /// Virtual function pointer table that customizes the behavior of this waker.
    vtable: &'static RawWakerVTable,
}

impl RawWaker {
    /// Creates a new `RawWaker` from the provided `data` pointer and `vtable`.
    ///
    /// The `vtable` customizes the behavior of a `Waker` which gets created
    /// from a `RawWaker`. For each operation on the `Waker`, the associated
    /// function in the `vtable` of the underlying `RawWaker` will be called.
    #[inline]
    #[rustc_promotable]
    #[stable(feature = "futures_api", since = "1.36.0")]
    #[rustc_const_stable(feature = "futures_api", since = "1.36.0")]
    pub const fn new(data: *const (), vtable: &'static RawWakerVTable) -> RawWaker {
        RawWaker { data, vtable }
    }
}
```

关于 `Waker`, 也就将 `RawWaker` 包装了起来，然后提供了一些方法

```rust
pub struct Waker {
    waker: RawWaker,
}

impl Waker {
    /// Wake up the task associated with this `Waker`.
    #[inline]
    #[stable(feature = "futures_api", since = "1.36.0")]
    pub fn wake(self) {
        // The actual wakeup call is delegated through a virtual function call
        // to the implementation which is defined by the executor.
        let wake = self.waker.vtable.wake;
        let data = self.waker.data;

        // Don't call `drop` -- the waker will be consumed by `wake`.
        crate::mem::forget(self);

        // SAFETY: This is safe because `Waker::from_raw` is the only way
        // to initialize `wake` and `data` requiring the user to acknowledge
        // that the contract of `RawWaker` is upheld.
        unsafe { (wake)(data) };
    }

    /// Wake up the task associated with this `Waker` without consuming the `Waker`.
    ///
    /// This is similar to `wake`, but may be slightly less efficient in the case
    /// where an owned `Waker` is available. This method should be preferred to
    /// calling `waker.clone().wake()`.
    #[inline]
    #[stable(feature = "futures_api", since = "1.36.0")]
    pub fn wake_by_ref(&self) {
        // The actual wakeup call is delegated through a virtual function call
        // to the implementation which is defined by the executor.

        // SAFETY: see `wake`
        unsafe { (self.waker.vtable.wake_by_ref)(self.waker.data) }
    }

    /// Returns `true` if this `Waker` and another `Waker` have awoken the same task.
    ///
    /// This function works on a best-effort basis, and may return false even
    /// when the `Waker`s would awaken the same task. However, if this function
    /// returns `true`, it is guaranteed that the `Waker`s will awaken the same task.
    ///
    /// This function is primarily used for optimization purposes.
    #[inline]
    #[stable(feature = "futures_api", since = "1.36.0")]
    pub fn will_wake(&self, other: &Waker) -> bool {
        self.waker == other.waker
    }

    /// Creates a new `Waker` from [`RawWaker`].
    ///
    /// The behavior of the returned `Waker` is undefined if the contract defined
    /// in [`RawWaker`]'s and [`RawWakerVTable`]'s documentation is not upheld.
    /// Therefore this method is unsafe.
    #[inline]
    #[stable(feature = "futures_api", since = "1.36.0")]
    pub unsafe fn from_raw(waker: RawWaker) -> Waker {
        Waker { waker }
    }
}

#[stable(feature = "futures_api", since = "1.36.0")]
impl Clone for Waker {
    #[inline]
    fn clone(&self) -> Self {
        Waker {
            // SAFETY: This is safe because `Waker::from_raw` is the only way
            // to initialize `clone` and `data` requiring the user to acknowledge
            // that the contract of [`RawWaker`] is upheld.
            waker: unsafe { (self.waker.vtable.clone)(self.waker.data) },
        }
    }
}

#[stable(feature = "futures_api", since = "1.36.0")]
impl Drop for Waker {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: This is safe because `Waker::from_raw` is the only way
        // to initialize `drop` and `data` requiring the user to acknowledge
        // that the contract of `RawWaker` is upheld.
        unsafe { (self.waker.vtable.drop)(self.waker.data) }
    }
}

#[stable(feature = "futures_api", since = "1.36.0")]
impl fmt::Debug for Waker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let vtable_ptr = self.waker.vtable as *const RawWakerVTable;
        f.debug_struct("Waker")
            .field("data", &self.waker.data)
            .field("vtable", &vtable_ptr)
            .finish()
    }
}
```
