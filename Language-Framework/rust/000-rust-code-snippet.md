
1. rust 多线程共享 raw pointer
https://www.reddit.com/r/rust/comments/hp1gd8/best_way_to_share_raw_pointers_or_unique/ 

```rust
/* rust中的 raw pointer 是不能Send。找了好久，找到一个 把指针转成 usize，然后在 线程操作时，再转回指针的方法.

试了 Fragile，不行
试了 struct Foo(*mut i32); unsafe impl Send for Foo{};   Foo(some_poniner). 不行。只要调用 .0 就会报错
*/

fn main() {
  let val = vec![1, 2, 3, 4];
  let val_ptr = val.as_mut_ptr() as usize;
  for i in 0..val.len() {
    thread::sparwn(move || {
        let ptr = val_ptr as *mut i32;
        *ptr.add(i) += 1;
    })
  }
}

```
