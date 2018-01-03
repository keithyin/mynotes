# GPU 编程： CUDA



## 如何组织自己的代码

* GPU 上跑的代码都写在 `.cu` 文件中， 
  * `__global__, __device__` 
* 然后写个 `launcher` 来调用 `kernel`
  * `launcher` **仅仅用来 调用 `kernel`**
  * 计算 `gridDim， blockDim`
  * 然后将 `launcher` 接口暴露给其它 应用就可以了。

**`.cu` 代码只需要搞定上面两个就可以了**



**剩下就是 `.cc/.c` 代码调用 `launcher` 了。  **

