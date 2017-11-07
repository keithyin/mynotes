# backward

```python
def backward(self, out_grad=None, retain_graph=False, train_mode=True):
    """Compute the gradients of this NDArray w.r.t variables.

    Parameters
    ----------
    out_grad : NDArray, optional
        Gradient with respect to head.
    retain_graph : bool, optional
        Whether to retain the computaion graph for another backward
        pass on the same graph. By default the computaion history
        is cleared.
    train_mode : bool, optional
        Whether to compute gradient for training or inference.
    """
    if out_grad is None:
        ograd_handles = [NDArrayHandle(0)]
    else:
        ograd_handles = [out_grad.handle]

    check_call(_LIB.MXAutogradBackwardEx(
        1, c_array(NDArrayHandle, [self.handle]),
        c_array(NDArrayHandle, ograd_handles),
        0,
        ctypes.c_void_p(0),
        ctypes.c_int(retain_graph),
        ctypes.c_int(0),
        ctypes.c_int(train_mode),
        ctypes.c_void_p(0),
        ctypes.c_void_p(0)))
```



```c
int MXAutogradBackwardEx(mx_uint num_output,
                         NDArrayHandle *output_handles,
                         NDArrayHandle *ograd_handles,
                         mx_uint num_variables,
                         NDArrayHandle *var_handles,
                         int retain_graph,
                         int create_graph,
                         int is_train,
                         NDArrayHandle **grad_handles,
                         int **grad_stypes) {
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();

  std::vector<NDArray*> outputs, ograds, variables;
  outputs.reserve(num_output);
  for (mx_uint i = 0; i < num_output; ++i) {
    outputs.emplace_back(reinterpret_cast<NDArray*>(output_handles[i]));
  }

  ograds.reserve(num_output);
  for (mx_uint i = 0; i < num_output; ++i) {
    if (ograd_handles != nullptr) {
      ograds.emplace_back(reinterpret_cast<NDArray*>(ograd_handles[i]));
    } else {
      ograds.emplace_back(nullptr);
    }
  }

  variables.reserve(num_variables);
  // num_variables = 0
  for (mx_uint i = 0; i < num_variables; ++i) {
    variables.emplace_back(reinterpret_cast<NDArray*>(var_handles[i]));
  }
  // backward 是用 Imperative 执行的 这个类才是大佬
  auto grads = Imperative::Get()->Backward(outputs, ograds, variables, is_train,
                                                  retain_graph, create_graph);
  if (num_variables != 0) {
    ret->ret_handles.clear();
    ret->out_types.clear();
    ret->ret_handles.reserve(grads.size());
    ret->out_types.reserve(grads.size());
    for (const auto& i : grads) {
      ret->ret_handles.push_back(i);
      ret->out_types.push_back(i->storage_type());
    }
    *grad_handles = dmlc::BeginPtr(ret->ret_handles);
    *grad_stypes = dmlc::BeginPtr(ret->out_types);
  }
  API_END();
}
```



```c
std::vector<NDArray*> Imperative::Backward(
    const std::vector<NDArray*>& outputs,
    const std::vector<NDArray*>& ograds,
    const std::vector<NDArray*>& variables,
    bool is_train, bool retain_graph,
    bool create_graph) {
  using namespace nnvm;
  using namespace imperative;
  static const std::vector<const Op*> zero_ops{Op::Get("zeros_like"), Op::Get("_zeros")};

  // Construct forward graph
  Graph graph;
  graph.outputs.reserve(outputs.size());
  for (const auto& i : outputs) {
    CHECK(!AGInfo::IsNone(*i))
      << "Cannot differentiate node because it is not in a computational graph. "
      << "You need to set is_recording to true or use autograd.record() to save "
      << "computational graphs for backward. If you want to differentiate the same "
      << "graph twice, you need to pass retain_graph=True to backward.";
    graph.outputs.emplace_back(i->entry_);
  }
  size_t num_forward_outputs = graph.outputs.size();

  // Prepare head gradients
  std::vector<NodeEntry> ograd_entries;
  ograd_entries.reserve(ograds.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    ograd_entries.emplace_back(NodeEntry{Node::Create(), 0, 0});
    AGInfo& info = AGInfo::Create(ograd_entries.back().node);
    info.ctx = outputs[i]->ctx();
    if (ograds[i] != nullptr) {
      info.outputs.emplace_back(*ograds[i]);
    } else {
      info.outputs.emplace_back(outputs[i]->shape(), outputs[i]->ctx(),
                                true, outputs[i]->dtype());
      info.outputs.back() = static_cast<real_t>(1.0);
    }
  }

  // Get gradient graph
  Symbol sym;
  sym.outputs = graph.outputs;
  std::vector<NodeEntry> xs;
  std::vector<NDArray*> x_grads;
  std::vector<OpReqType> x_reqs;
  if (variables.size()) {
    xs.reserve(variables.size());
    x_grads.reserve(variables.size());
    x_reqs.reserve(variables.size());
    for (size_t i = 0; i < variables.size(); ++i) {
      CHECK(!AGInfo::IsNone(*variables[i]) &&
            AGInfo::IsVariable(variables[i]->entry_.node))
          << "Cannot differentiate with respect to the " << i+1 << "-th variable"
          << " because it does not require gradient.";
      xs.emplace_back(variables[i]->entry_);
      x_grads.push_back(new NDArray());
      x_reqs.push_back(kWriteTo);
    }
  } else {
    std::vector<NodePtr> args = sym.ListInputs(Symbol::kReadOnlyArgs);
    xs.reserve(args.size());
    x_grads.reserve(args.size());
    x_reqs.reserve(args.size());
    for (const auto& i : args) {
      AGInfo& info = AGInfo::Get(i);
      if (info.grad_req == kNullOp) continue;
      xs.emplace_back(NodeEntry{i, 0, 0});
      x_grads.push_back(&info.out_grads[0]);
      x_reqs.push_back(info.grad_req);
      info.fresh_out_grad = true;
    }
    CHECK_GT(xs.size(), 0)
        << "There are no inputs in computation graph that require gradients.";
  }

  Graph g_graph = pass::Gradient(
      graph, graph.outputs, xs, ograd_entries,
      exec::AggregateGradient, nullptr, nullptr,
      zero_ops, "_copy");
  CHECK_EQ(g_graph.outputs.size(), xs.size());
  for (const auto &e : g_graph.outputs) {
    graph.outputs.push_back(e);
  }
  const auto& idx = graph.indexed_graph();
  // get number of nodes used in forward pass
  size_t num_forward_nodes = 0;
  size_t num_forward_entries = 0;
  for (size_t i = 0; i < num_forward_outputs; ++i) {
    num_forward_nodes = std::max(
        num_forward_nodes, static_cast<size_t>(idx.outputs()[i].node_id + 1));
    num_forward_entries = std::max(
        num_forward_entries, static_cast<size_t>(idx.entry_id(idx.outputs()[i])) + 1);
  }

  // Allocate buffer
  std::vector<NDArray> buff(idx.num_node_entries());
  std::vector<uint32_t> ref_count(buff.size(), 0);
  std::vector<OpStatePtr> states;
  std::vector<NDArray*> arrays;
  arrays.reserve(buff.size());
  for (size_t i = 0; i < buff.size(); ++i) arrays.push_back(&buff[i]);
  if (create_graph) {
    states.resize(num_forward_nodes);
    nnvm::DFSVisit(sym.outputs, [&](const nnvm::NodePtr& n) {
      AGInfo& info = AGInfo::Get(n);
      states[idx.node_id(n.get())] = info.state;
      for (uint32_t i = 0; i < info.outputs.size(); ++i) {
        CHECK(idx.exist(n.get()));
        size_t nid = idx.node_id(n.get());
        size_t eid = idx.entry_id(nid, i);
        buff[eid] = info.outputs[i];
        buff[eid].entry_ = NodeEntry{n, i, 0};
        ref_count[eid] = 1;
      }
    });
    for (size_t i = 0; i < ograd_entries.size(); ++i) {
      AGInfo& info = AGInfo::Get(ograd_entries[i].node);
      if (!idx.exist(ograd_entries[i].node.get())) continue;
      size_t eid = idx.entry_id(ograd_entries[i]);
      buff[eid] = info.outputs[0];
      buff[eid].entry_ = ograd_entries[i];
    }
  } else {
    states.reserve(num_forward_nodes);
    for (size_t i = 0; i < num_forward_nodes; ++i) {
      const AGInfo& info = dmlc::get<AGInfo>(idx[i].source->info);
      states.emplace_back(info.state);
      for (size_t j = 0; j < info.outputs.size(); ++j) {
        size_t eid = idx.entry_id(i, j);
        arrays[eid] = const_cast<NDArray*>(&(info.outputs[j]));

        if (retain_graph || info.grad_req != kNullOp) ref_count[eid] = 1;
      }
    }
    for (size_t i = 0; i < ograd_entries.size(); ++i) {
      if (!idx.exist(ograd_entries[i].node.get())) continue;
      AGInfo& info = AGInfo::Get(ograd_entries[i].node);
      arrays[idx.entry_id(ograd_entries[i])] = &info.outputs[0];
    }
  }
  for (size_t i = num_forward_outputs; i < graph.outputs.size(); ++i) {
    size_t eid = idx.entry_id(graph.outputs[i]);
    arrays[eid] = x_grads[i - num_forward_outputs];
    ref_count[eid] = 1;
  }

  // Assign context
  auto vctx = PlaceDevice(idx);

  // Infer shape type
  {
    std::pair<uint32_t, uint32_t> node_range, entry_range;
    node_range = {num_forward_nodes, idx.num_nodes()};
    entry_range = {num_forward_entries, idx.num_node_entries()};

    ShapeVector shapes;
    shapes.reserve(idx.num_node_entries());
    for (const auto& i : arrays) shapes.emplace_back(i->shape());
    CheckAndInferShape(&graph, std::move(shapes), false,
                       node_range, entry_range);

    DTypeVector dtypes;
    dtypes.reserve(idx.num_node_entries());
    for (const auto& i : arrays) dtypes.emplace_back(i->dtype());
    CheckAndInferType(&graph, std::move(dtypes), false,
                      node_range, entry_range);

    StorageTypeVector stypes;
    stypes.reserve(idx.num_node_entries());
    for (const auto& i : arrays) stypes.emplace_back(i->storage_type());
    exec::DevMaskVector dev_mask;
    dev_mask.reserve(idx.num_nodes());
    for (const auto& i : vctx) dev_mask.emplace_back(i.dev_mask());
    CheckAndInferStorageType(&graph, std::move(dev_mask), std::move(stypes), false,
                             node_range, entry_range);
  }

  // Calculate ref count
  for (size_t i = num_forward_nodes; i < idx.num_nodes(); ++i) {
    for (const auto& j : idx[i].inputs) {
       ++ref_count[idx.entry_id(j)];
    }
  }

  // Assign reqs
  std::vector<OpReqType> array_reqs(arrays.size(), kWriteTo);
  for (size_t i = num_forward_entries; i < idx.num_node_entries(); ++i) {
    if (ref_count[i] == 0) array_reqs[i] = kNullOp;
  }
  for (size_t i = num_forward_outputs; i < idx.outputs().size(); ++i) {
    size_t eid = idx.entry_id(idx.outputs()[i]);
    array_reqs[eid] = x_reqs[i - num_forward_outputs];
  }

  const auto& shapes = graph.GetAttr<ShapeVector>("shape");
  const auto& dtypes = graph.GetAttr<DTypeVector>("dtype");
  const auto& stypes = graph.GetAttr<StorageTypeVector>("storage_type");
  const auto& dispatch_modes = graph.GetAttr<DispatchModeVector>("dispatch_mode");

  for (size_t i = num_forward_nodes; i < idx.num_nodes(); ++i) {
    auto num_outputs = idx[i].source->num_outputs();
    for (size_t j = 0; j < num_outputs; ++j) {
      auto eid = idx.entry_id(i, j);
      if (!arrays[eid]->is_none()) continue;
      if (stypes[eid] == kDefaultStorage) {
        *arrays[eid] = NDArray(shapes[eid], vctx[i], true, dtypes[eid]);
      } else {
        *arrays[eid] = NDArray(static_cast<NDArrayStorageType>(stypes[eid]),
                               shapes[eid], vctx[i], true, dtypes[eid]);
      }
    }
  }

  // Execution

  bool prev_recording = set_is_recording(create_graph);
  bool prev_training = set_is_training(is_train);

  RunGraph(retain_graph, idx, arrays, num_forward_nodes, idx.num_nodes(),
           std::move(array_reqs), std::move(ref_count), &states, dispatch_modes);

  set_is_recording(prev_recording);
  set_is_training(prev_training);

  // Clear history
  if (!retain_graph) {
    nnvm::DFSVisit(sym.outputs, [&](const nnvm::NodePtr& n) {
      AGInfo::Clear(n);
      n->inputs.clear();
    });
  }

  if (variables.size()) {
    return x_grads;
  }
  return {};
}
```





## 参考资料

[https://github.com/apache/incubator-mxnet/blob/master/src/c_api/c_api_ndarray.cc](https://github.com/apache/incubator-mxnet/blob/master/src/c_api/c_api_ndarray.cc)

[https://github.com/apache/incubator-mxnet/blob/master/src/c_api/c_api_ndarray.cc#L301](https://github.com/apache/incubator-mxnet/blob/master/src/c_api/c_api_ndarray.cc#L301)

