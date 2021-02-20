# `use std::ops::Deref`

> 引用相关？

* Used for immutable dereferencing operations, like *v.
* Deref coercion.

```rust
use std::ops::Deref;

struct DerefExample<T> {
    value: T
}

impl<T> Deref for DerefExample<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

let x = DerefExample { value: 'a' };
assert_eq!('a', *x);
```

# `std::ops::DerefMut`

> 借用相关？

* Used for mutable dereferencing operations, like in *v = 1;
* Deref coercion

```rust
use std::ops::{Deref, DerefMut};

struct DerefMutExample<T> {
    value: T
}

impl<T> Deref for DerefMutExample<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> DerefMut for DerefMutExample<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

let mut x = DerefMutExample { value: 'a' };
*x = 'b';
assert_eq!('b', *x);
```

# `std::marker::Copy`

* 将一个类型标记为 Copy, 说明其
let mut x = DerefMutExample { value: 'a' }; 
