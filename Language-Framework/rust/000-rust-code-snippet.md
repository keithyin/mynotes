
1. rust 多线程共享 raw pointer
https://www.reddit.com/r/rust/comments/hp1gd8/best_way_to_share_raw_pointers_or_unique/ 

```rust
/* rust中的 raw pointer 是不能Send。找了好久，找到俩方法
1. 把指针转成 usize，然后在 线程操作时，再转回指针的方法.
2. 给指针 impl Send

试了 Fragile，不行。
*/

#[derive(Debug)]
struct SendableI32Pointer(*mut i32);
unsafe impl Send for SendableI32Pointer {}

fn main() {
  let val = vec![1, 2, 3, 4];
  let val_ptr = val.as_mut_ptr() as usize;
  for i in 0..val.len() {
    thread::sparwn(move || {
        let ptr = val_ptr as *mut i32;
        *ptr.add(i) += 1;
    })
  }

  // 给 raw pointer 实现 Send。因为没办法直接 unsafe impl Send for *mut i32. 只能包一下，再 unsafe impl Send ..
  for i in 0..val.len() {
      let ptr = SendableI32Pointer(val.as_mut_ptr());
      thread::sparwn(move || {
          let v = &ptr; // 这句似乎得留着，不然可能还会报错。奇怪？难道是编译器优化的问题？
          let raw_ptr = ptr.0;
          *raw_ptr.add(i) += 1;
      })
    }

}

```
