# pytorch 源码阅读：autograd

博主水平有限，如有错误，请不吝指出。

[pytorch源码注释，欢迎 pr，提 issue 和 star](https://github.com/KeithYin/pytorch)



当我们使用 `pytorch` 的 `python` 的接口编写代码的时候，感觉是十分清爽的，不需要考虑底层的实现。但是好奇心驱使我们 想一探究竟，看看底层 `C/C++` 那部分到底做了什么。

本篇文章主要专注于：

* `pytorch` 是如何动态构建反向传导图的
* `pytorch` 的反向传导是怎么操作的



## pytorch 是如何构建反向传导图的

![](../../imgs/dynamic_graph.gif)

这是 `pytorch` 官方的一张图，第一次看到这个图，感觉很奇怪，怎么箭头指向的并不是 tensor 流动方向呢（对比 tensorflow观望的那张图）？到最后读了源码才发现，**原来 `pytorch` 实际上是在 动态 构建一个 反向传导计算图！！**这张图很直白的表达除了 `pytorch` 的底层思想。



那么 `pytorch` 是如何动态构建 反向传导计算图的呢？ [先看一部分代码](https://github.com/pytorch/pytorch/blob/v0.2.0/torch/csrc/autograd/functions/basic_ops.cpp#L24)

```c++
// Add 函数 计算方法的实现， 构建反向传导图的关键在 wrap_outputs
auto Add::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("Add", inputs, 2);
  auto& input1 = inputs[0]->data;
  auto& input2 = inputs[1]->data;
  AutoGPU guard(input1->getDevice());

  bool first_sparse = input1->isSparse();
  auto output = first_sparse ? input2->newTensor() : input1->newTensor();
  if (first_sparse) {
    output->cadd(*input2, *input1);
  } else {
    output->cadd(*input1, *input2);
  }

  return wrap_outputs(inputs, as_tensor_list(std::move(output)), [&](FunctionFlags f) {
    return std::make_shared<AddBackward>(std::move(f));
  });
};
```

动态构建 反向传导 计算图的 核心代码是 `wrap_outputs`. 它做的事情有：

* 根据 forward 过程中的 inputs 来计算 backward 函数的 flag （is_volatile, is_executable, next_functions）
* 然后将 forward 的输出 的 grad_fn 设置成 创建好的 backward 函数。
* 这样，函数节点就构成了一张 反向传导图！（通过不停的 .next_functions.next_functions） 

下面是代码

```c++
variable_list wrap_outputs(const variable_list& inputs, tensor_list&& outputs,
                           function_constructor ctr) {

  // 使用 inputs variables 来计算 反向传导的 Function 的 flag (f.is_executable, f.is_volatile)和 next_functions
  // 这里需要搞清楚的一点是：inputs 是前向传导的 inputs，从它可获得的信息有：当前 函数的 反向传导函数 是否 可执行，

  auto flags = Function::flags(inputs);
  variable_list result;

  // 开始创建 返回的 Variable 了。
  result.reserve(outputs.size());
  
  if (flags.is_volatile) {  // 如果 is_volatile=true, 那么输出的 Variable 的 is_volatile=true 
    for (auto& output : outputs) {
      if (output.defined()) { // 因为 可能返回 None 嘛，所以这里 check 一下
        result.emplace_back(make_variable(output, false, true)); // requires_grad=false, is_volatile=true
      } else {
        result.emplace_back(); 
      }
    }
  } else {  // 如果 volatile=false， 难道也不管 is_executable 了吗？ 
    // ctr 是一个 lambda 函数， 它返回一个 std::shared_ptr<GradFn>
    // 梯度 使用 Function::flags 计算出来的 flags 其实是给 Backward 用的。
    auto grad_fn = ctr(std::move(flags));  // 用 flags(is_executable, is_volatile) 创建出来一个 Function。
    for (auto& output : outputs) {
      if (output.defined()) {
        result.emplace_back(make_variable(output, grad_fn));
      } else {
        // forward 的输出 变量个数，就是 backward 的输入变量个数。
        ++grad_fn->num_inputs;
        result.emplace_back();
      }
    }
  }
  return result;
}
```



## pytorch 的反向传导计算过程

反向传导时要考虑的第一个问题就是：

```python
a = o+e
c = a+b
d = a+e
res = c+d
```

假设 `res` 对 `o` 求导。只有求对了 `a` 的梯度，`o`的梯度才能正确求出。 `a`  的梯度来自于两 条路径，一个是 `d`，一个是 `e`。`backward` 过程要保证的到正确的 `a` 梯度。因为 `pytorch` 是通过 `function` 节点 构建出来的一个反向传导图， 所以将这个问题看作 求 grad_fn 的 **`输入`**问题， `pytorch ` 解决这个问题的思路是：

1.  创建了一个新的 结构体 `FunctionTask`， 里面有个 `InputBuffer` 属性，这个是用来累积来自不同路径的梯度的
2. 什么时候才累积完呢？ pytorch 对每个 grad_fun 节点都求了其依赖 , 比如 上例中的 `grad_fn(a,o,e)` 的依赖就是 2, 因为，`a` 被用了两次。 `grad_fn(a,o,e)` 没聚集一次梯度，其依赖就 -1, 当依赖为 0 的时候，就将其对应的 `FunctionTask` 放到 `ready_queue` 中等待 被执行。



**等到 ready_queue 中没有 FunctionTask 了，backward过程也就完成了**

[详细代码](https://github.com/KeithYin/pytorch/tree/master/torch/csrc/autograd)

`backward` 过程用到的一些 数据结构

```c++
struct FunctionTask {
  // 每个 FunctionTask 中都维护着一个 base GraphTask
  GraphTask* base;
  std::shared_ptr<Function> fn; // 代表 grad_fn
  InputBuffer inputs; // 累积 grad_fn 的输入

  FunctionTask(GraphTask* base, std::shared_ptr<Function> fn, InputBuffer inputs)
    : base(base)
    , fn(fn)
    , inputs(std::move(inputs)) {}
};
```

```c++
struct ReadyQueue {
  // 用来 存放可被 执行的 FunctionTask
  // queue 是 FunctionTask 的 一个 双端队列
  std::deque<FunctionTask> queue;
  // std::condition_variable 条件变量，同步的时候会用到
  // 用 unique_lock (over mutex) 来进行操作
  std::condition_variable not_empty;
  std::mutex mutex;

  void push_front(FunctionTask item);
  FunctionTask pop_back();
};
```



```c++
struct GraphTask {
  // 记录整个反向计算图的依赖情况 等等。
  std::exception_ptr exception;
  // Indicates if an error occurred while executing any task.  When this is
  // true, it signals all threads to stop executing.
  std::atomic_bool has_error;

  // 剩余 tasks。 在 ReadyQueue 的 push方法 中加一， 在 evaluate_function 中减一操作
  std::atomic<uint64_t> outstanding_tasks;
  bool keep_graph;
  bool has_any_work;
  // 用来 给 notify_all 加锁的
  std::mutex mutex;
  // Notified when a task finishes executing.  Check outstanding_tasks to see
  // if all tasks are done.
  std::condition_variable not_done;
  const Engine::pre_callback_map& pre_callbacks;
  const Engine::post_callback_map& post_callbacks;

  //用来存放 没有 准备好的 FunctionTask
  std::unordered_map<Function*, InputBuffer> not_ready;
  // 记录 所有 Function 节点的 依赖
  std::unordered_map<Function*, int> dependencies;
  
  // 这个来 表示 GraphTask 是在哪个 device 上创建的
  int owner;

};
```



```c++
struct Engine{
  // 反向传导计算引擎
}
```







