# __thread 关键字，Thread Local Storage

* 三种使用形式
  * `__thread int i;`
  * `extern __thread struct state s;`
  * `static __thread char *p;`
* 使用的位置
  * `global`
  * `file-scoped static`
  * `function-scoped static`
  * `static data member of class`
* 不能用在的位置
  * `block-scoped automatic`
  * `non-static data member`
* 注意事项
  * In C++, if an initializer is present for a thread-local variable, it must be a constant-expression



## 参考资料

[https://gcc.gnu.org/onlinedocs/gcc-3.4.6/gcc/Thread_002dLocal.html](https://gcc.gnu.org/onlinedocs/gcc-3.4.6/gcc/Thread_002dLocal.html)

[https://stackoverflow.com/questions/6317673/why-to-use-thread-local-storage-tlsalloc-tlsgetvalue-ets-instead-of-local-va](https://stackoverflow.com/questions/6317673/why-to-use-thread-local-storage-tlsalloc-tlsgetvalue-ets-instead-of-local-va)

