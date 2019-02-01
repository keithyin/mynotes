# 反向求导引擎



**几个重要的数据结构**

* `FunctionTask`： 将 `grad_fn` 封装成 `FunctionTask`，增加了 `InputBuffer`属性，用于累积 `grad_fn` 的输入梯度。

```c++
struct FunctionTask {
  // function task 所 属 的 GraphTask
  GraphTask* base;
  std::shared_ptr<Function> fn;
  // This buffer serves as an implicit "addition" node for all of the
  // gradients flowing here.  Once all the dependencies are finished, we
  // use the contents of this buffer to run the function.
  InputBuffer inputs;

  FunctionTask(GraphTask* base, std::shared_ptr<Function> fn, InputBuffer inputs)
    : base(base)
    , fn(std::move(fn))
    , inputs(std::move(inputs)) {}
};
```



* `ReadyQueue`：用来存放可以被引擎计算的 `FucntionTask` 的地方。反向求导引擎实现为一个 生产者-消费者的模式，`ReadyQueue` 就是两者的共享空间。
  * 生产者：用来生产 `FunctionTask`，放到 `ReadyQueue` 中。
  * 消费者：用来消费 `ReadyQueue` 中的 `FunctionTask`

```c++
struct ReadyQueue {
  std::priority_queue<FunctionTask, std::vector<FunctionTask>, CompareFunctionTaskTime> heap;
  
  // 生产者消费者所需要的 condition_variable 和 mutex
  std::condition_variable not_empty;
  std::mutex mutex;

  void push(FunctionTask item);
  FunctionTask pop();
};
```



* `GraphTask` :  holds metadata needed for **a single execution** of backward()
  * 每次 `.backward()` 都会创建一个 `GraphTask`
  * 当前进程只有一个 `Engine` 实例

```c++
// GraphTask holds metadata needed for a single execution of backward()
struct GraphTask {
  
  std::atomic<uint64_t> outstanding_tasks;
  bool keep_graph;
  bool grad_mode;

  std::mutex mutex;
  // Notified when a task finishes executing.  Check outstanding_tasks to see
  // if all tasks are done.
  std::condition_variable not_done;
  std::unordered_map<Function*, InputBuffer> not_ready;
  std::unordered_map<Function*, int> dependencies;

  void init_to_execute(Function& graph_root, const edge_list& outputs);

  // The value of worker_device in the thread that created this task.
  // See Note [Reentrant backwards]
  int owner;

  bool can_checkpoint() {
    return exec_info.empty();
  }

  GraphTask(bool keep_graph, bool grad_mode)
    : has_error(false)
    , outstanding_tasks(0)
    , keep_graph(keep_graph)
    , grad_mode(grad_mode)
    , owner(NO_DEVICE) {}
};
```



* `Engine` : 反向求导引擎，一个进程中只有 **一个实例**

```c++
struct TORCH_API Engine {
  /// Returns a reference to a static `Engine` instance.
  static Engine& get_default_engine();

  Engine();
  virtual ~Engine();

  using ready_queue_type = std::deque<std::pair<std::shared_ptr<Function>, InputBuffer>>;
  using dependencies_type = std::unordered_map<Function*, int>;

  // Given a list of (Function, input number) pairs computes the value of the graph
  // by following next_edge references.
  virtual variable_list execute(
      const edge_list& roots,
      const variable_list& inputs,
      bool keep_graph,
      bool create_graph,
      const edge_list& outputs = {});
  virtual std::unique_ptr<AnomalyMetadata> make_anomaly_metadata() {
    return nullptr;
  }

  void queue_callback(std::function<void()> callback);

  bool is_checkpoint_valid();

protected:
  void compute_dependencies(Function* root, GraphTask& task);
  void evaluate_function(FunctionTask& task);
  ReadyQueue& ready_queue(int device);
  void start_threads();
  virtual void thread_init(int device);
  virtual void thread_main(GraphTask *graph_task);
  virtual void thread_on_exception(FunctionTask& task, std::exception& e);

  std::once_flag start_threads_flag;
  std::vector<std::shared_ptr<ReadyQueue>> ready_queues;
  std::vector<std::function<void()>> final_callbacks;
  std::mutex post_callbacks_lock;
};
```



## Engine

```c++
void Variable::backward(
    c10::optional<Tensor> gradient,
    bool keep_graph,
    bool create_graph) const {
  auto autograd_meta = get_autograd_meta();
  std::vector<Edge> edges;
  
  // output_nr_ : 当前variable是 fn 的第几个输出，如果是第2个，output_nr_=1.
  edges.emplace_back(autograd_meta->grad_fn_, autograd_meta->output_nr_);

  std::vector<Variable> inputs;
  if (!gradient.has_value()) {
    gradient = make_variable(at::ones_like(data()), /*requires_grad=*/false);
  }
  inputs.push_back(std::move(as_variable_ref(*gradient)));
  Engine::get_default_engine().execute(edges, inputs, keep_graph, create_graph);
}
```



```c++
auto Engine::execute(const edge_list& roots,
                     const variable_list& inputs,
                     bool keep_graph,
                     bool create_graph,
                     const edge_list& outputs) -> variable_list {
  
  /* 开 1+num_gpu 个 worker thread 准备操作, 
     等着ReadyQueue中有FunctionTask，worker thread就开始操作。
  */  
  std::call_once(start_threads_flag, &Engine::start_threads, this);

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  validate_outputs(roots, const_cast<variable_list&>(inputs), [](const std::string& msg) {
    return msg;
  });

  // Callbacks are only valid for the duration of this run and should always be cleared
  ClearCallbacks _cb_guard(final_callbacks, post_callbacks_lock);

  GraphTask graph_task(keep_graph, create_graph);
  std::unique_lock<std::mutex> lock(graph_task.mutex);

  /* Now compute the dependencies for all executable functions and queue the root
  	 为什么需要GraphRoot：
  	 	- 反向传导时，grad_fn为图中的节点，但是最后一个variable可能有一个输入梯度
  	 	- Graph 就将 输入梯度封装起来，然后正确设置 next_fn_ 就可以了。
  */
  auto graph_root = std::make_shared<GraphRoot>(roots, inputs);
  
  /* 依赖是 Function 的依赖，只有当 dependencies=0时，说明，输入梯度已经正确的累积了
  	这时候就可以计算当前的 grad_fn 了。
  */
  compute_dependencies(graph_root.get(), graph_task);
  if (!outputs.empty()) {
    graph_task.init_to_execute(*graph_root, outputs);
  }
  
  // 将 GraphRoot 对应的 FunctionTask 放到 cpu 的ReadyQueue 中，然后就执行起来了
  ready_queue(-1).push(FunctionTask(&graph_task, std::move(graph_root), InputBuffer(0)));

  // Not a worker thread.
  if (worker_device == NO_DEVICE) {
    // Wait for all tasks to complete
    graph_task.not_done.wait(lock, [&graph_task]{
      return graph_task.outstanding_tasks.load() == 0;
    });
  } else {
    // Get back to work while we wait for our new graph_task to
    // complete!
    // See Note [Reentrant backwards]
    graph_task.owner = worker_device;
    lock.unlock();
    thread_main(&graph_task);
  }
```



```c++

auto Engine::thread_main(GraphTask *graph_task) -> void {
  auto queue = ready_queues[worker_device + 1];
  // Why the test on graph_task->outstanding_tasks?  See
  // Note [Reentrant backwards]
  while (!graph_task || graph_task->outstanding_tasks > 0) {
    FunctionTask task = queue->pop();
    if (task.fn && !task.base->has_error.load()) {
      GradMode::set_enabled(task.base->grad_mode);
      try {
        evaluate_function(task);
      } catch (std::exception& e) {
        thread_on_exception(task, e);
      }
    }
    // Notify downstream about the completion of tasks depending
    // on both where the task was executed, and who owned the overall
    // graph (in case of reentrant execution.)  See Note [Reentrant backwards].
    auto base_owner = task.base->owner;
    // GraphTask from a non-worker thread（最基本的情况）. Easy case.
    if (base_owner == NO_DEVICE) {
      if (--task.base->outstanding_tasks == 0) {
        std::lock_guard<std::mutex> lock(task.base->mutex);
        task.base->not_done.notify_all();
      }
    } else {
      // If it's a task initiated from this thread, decrease the counter, but
      // don't do anything - loop condition will do all checks for us next.
      if (base_owner == worker_device) {
        --task.base->outstanding_tasks;
      // Otherwise send a dummy function task to the owning thread just to
      // ensure that it's not sleeping. If it has work, it might see that
      // graph_task->outstanding_tasks == 0 before it gets to the task, but
      // it's a no-op anyway.
      } else if (base_owner != worker_device) {
        if (--task.base->outstanding_tasks == 0) {
          // Synchronize outstanding_tasks with queue mutex
          std::atomic_thread_fence(std::memory_order_release);
          ready_queue(base_owner).push(FunctionTask(task.base, nullptr, InputBuffer(0)));
        }
      }
    }
  }
}
```





```c++
auto Engine::evaluate_function(FunctionTask& task) -> void {
  // If exec_info is not empty, we have to instrument the execution
  auto & exec_info = task.base->exec_info;
  if (!exec_info.empty()) {
    auto & fn_info = exec_info.at(task.fn.get());
    if (auto *capture_vec = fn_info.captures.get()) {
      std::lock_guard<std::mutex> lock(task.base->mutex);
      for (auto capture : *capture_vec) {
        task.base->captured_vars[capture.output_idx] = task.inputs[capture.input_idx];
      }
    }
    if (!fn_info.needed) return;
  }

  auto outputs = call_function(task);

  auto& fn = *task.fn;
  if (!task.base->keep_graph) {
    fn.release_variables();
  }

  int num_outputs = outputs.size();
  if (num_outputs == 0) return; // Don't even acquire the mutex

  if (AnomalyMode::is_enabled()) {
    AutoGradMode grad_mode(false);
    for (int i = 0; i < num_outputs; ++i) {
      auto& output = outputs[i];
      at::OptionalDeviceGuard guard(device_of(output));
      if (output.defined() && output.ne(output).any().item<uint8_t>()) {
        std::stringstream ss;
        ss << "Function '" << fn.name() << "' returned nan values in its " << i << "th output.";
        throw std::runtime_error(ss.str());
      }
    }
  }

  std::lock_guard<std::mutex> lock(task.base->mutex);
  for (int i = 0; i < num_outputs; ++i) {
    auto& output = outputs[i];
    const auto& next = fn.next_edge(i);

    if (!next.is_valid()) continue;

    // Check if the next function is ready to be computed
    bool is_ready = false;
    auto& dependencies = task.base->dependencies;
    auto it = dependencies.find(next.function.get());
    if (it == dependencies.end()) {
      auto name = next.function->name();
      throw std::runtime_error(std::string("dependency not found for ") + name);
    } else if (--it->second == 0) {
      dependencies.erase(it);
      is_ready = true;
    }

    auto& not_ready = task.base->not_ready;
    auto not_ready_it = not_ready.find(next.function.get());
    if (not_ready_it == not_ready.end()) {
      // Skip functions that aren't supposed to be executed
      if (!exec_info.empty()) {
        auto it = exec_info.find(next.function.get());
        if (it == exec_info.end() || !it->second.should_execute()) {
          continue;
        }
      }
      // No buffers have been allocated for the function
      InputBuffer input_buffer(next.function->num_inputs());
      input_buffer.add(next.input_nr, std::move(output));
      if (is_ready) {
        auto& queue = ready_queue(input_buffer.device());
        queue.push(FunctionTask(task.base, next.function, std::move(input_buffer)));
      } else {
        not_ready.emplace(next.function.get(), std::move(input_buffer));
      }
    } else {
      // The function already has a buffer
      auto &input_buffer = not_ready_it->second;
      input_buffer.add(next.input_nr, std::move(output));
      if (is_ready) {
        auto& queue = ready_queue(input_buffer.device());
        queue.push(FunctionTask(task.base, next.function, std::move(input_buffer)));
        not_ready.erase(not_ready_it);
      }
    }
  }
}
```



## Function







## next step

* 搞清楚 Function 和 Edge 还有 Tensor Variable 之间的关系

