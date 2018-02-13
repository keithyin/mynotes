# Layer 类源码分析

当我们想要实现一个层的时候，可以通过继承 `Layer` 类来实现：

* 重写 `__init__` 方法，用来保存 层的配置信息

* 重写 `build()` 方法： `__call__` 方法会调用这个方法

  * 里面应该调用 `add_variable` 
  * 然后调用 `super's build()` 方法，它会设置 `self.built=True`
* 重写 `call()` 方法： 执行 前向计算操作。





```python
class Layer(object):
  """Base layer class.

  This is the class from which all layers inherit, implementing common
  infrastructure functionality.

  Layer： managing variables,losses, and updates, as well as applying TensorFlow ops to input tensors.

  Users will just instantiate it and then treat it as a callable.

  We recommend that descendants of Layer implement the following methods:
  * `__init__()`: Save configuration in member variables
  * `build()`: Called once from `__call__`, when we know the shapes of inputs
    and `dtype`. Should have the calls to `add_variable()`, and then
    call the super's `build()` (which sets `self.built = True`, which is
    nice in case the user wants to call `build()` manually before the
    first `__call__`).
  * `call()`: Called in `__call__` after making sure `build()` has been called
    once. Should actually perform the logic of applying the layer to the
    input tensors (which should be passed in as the first argument).

  Read-only properties:
    `name`: The name of the layer (string).
    `dtype`: Default dtype of the layer (default of `None` means use the
      type of the first input).
    `trainable_variables`: List of trainable variables.
    `non_trainable_variables`: List of non-trainable variables.
    `variables`: List of all variables of this layer, trainable and
      non-trainable.
    `updates`: List of update ops of this layer.
    `losses`: List of losses added by this layer.

  Mutable properties:
    `trainable`: Whether the layer should be trained (boolean).
    `input_spec`: Optional (list of) `InputSpec` object(s) specifying the
      constraints on inputs that can be accepted by the layer.
  """

  def __init__(self, trainable=True, name=None, dtype=None,
               activity_regularizer=None, **kwargs):
    # We use a kwargs dict here because these kwargs only exist
    # for compatibility reasons.
    # The list of kwargs is subject to changes in the future.
    # We do not want to commit to it or to expose the list to users at all.
    # Note this is exactly as safe as defining kwargs in the function signature,
    # the only difference being that the list of valid kwargs is defined
    # below rather rather in the signature, and default values are defined
    # in calls to kwargs.get().
    allowed_kwargs = {
        '_scope',
        '_reuse',
        'input_shape',  # For compatibility with Keras `Sequential` model.
        'batch_size',  # For compatibility with Keras `Sequential` model.
    }
    for kwarg in kwargs:
      if kwarg not in allowed_kwargs:
        raise TypeError('Keyword argument not understood:', kwarg)

    # Mutable properties
    self.trainable = trainable
    self.built = False
    self.input_spec = None

    if activity_regularizer and context.in_eager_mode():
      raise ValueError(
          ('Activity regularization is not supported when executing eagerly. '
           'Got activity_regularizer=%s') % (activity_regularizer,))
    self._activity_regularizer = activity_regularizer
    self._trainable_weights = []
    self._non_trainable_weights = []
    self._updates = []
    # When executing eagerly, _losses is a list of zero-argument lambdas which
    # return tensors. When using graph execution, _losses is a list of ops.
    self._losses = []
    self._reuse = kwargs.get('_reuse')
    self._graph = ops.get_default_graph()
    self._per_input_losses = {}
    self._per_input_updates = {}
    self._dtype = None if dtype is None else dtypes.as_dtype(dtype).name
    call_fn_args = estimator_util.fn_args(self.call)
    self._compute_previous_mask = ('mask' in call_fn_args or
                                   hasattr(self, 'compute_mask'))
    self._call_has_scope_arg = 'scope' in call_fn_args

    # These lists will be filled via successive calls
    # to self._add_inbound_node().
    self._inbound_nodes = []
    self._outbound_nodes = []

    self._init_set_name(name)

    # Determine variable scope.
    scope = kwargs.get('_scope')
    if scope:
      with vs.variable_scope(scope) as captured_scope:
        self._scope = captured_scope
    else:
      self._scope = None

    # Set `_batch_input_shape` attribute
    # for compatibility with Keras `Sequential` model.
    if 'input_shape' in kwargs:
      batch_size = kwargs.get('batch_size')
      self._batch_input_shape = (batch_size,) + tuple(kwargs['input_shape'])

  def _init_set_name(self, name):
    # Determine layer name (non-unique).
    if isinstance(name, vs.VariableScope):
      base_name = name.name
    else:
      base_name = name
      self._name = name
    if not name:
      self._name, base_name = self._make_unique_name()
    self._base_name = base_name
  
  def build(self, _):
    """Creates the variables of the layer."""
    self.built = True

  def call(self, inputs, **kwargs):  # pylint: disable=unused-argument
    """The logic of the layer lives here.

    Arguments:
      inputs: input tensor(s).
      **kwargs: additional keyword arguments.

    Returns:
      Output tensor(s).
    """
    return inputs

  def add_variable(self, name, shape, dtype=None,
                   initializer=None, regularizer=None,
                   trainable=True, constraint=None,
                   partitioner=None):
    """Adds a new variable to the layer, or gets an existing one; returns it.

    Arguments:
      name: variable name.
      shape: variable shape.
      dtype: The type of the variable. Defaults to `self.dtype` or `float32`.
      initializer: initializer instance (callable).
      regularizer: regularizer instance (callable).
      trainable: whether the variable should be part of the layer's
        "trainable_variables" (e.g. variables, biases)
        or "non_trainable_variables" (e.g. BatchNorm mean, stddev).
        Note, if the current variable scope is marked as non-trainable
        then this parameter is ignored and any added variables are also
        marked as non-trainable.
      constraint: constraint instance (callable).
      partitioner: (optional) partitioner instance (callable).  If
        provided, when the requested variable is created it will be split
        into multiple partitions according to `partitioner`.  In this case,
        an instance of `PartitionedVariable` is returned.  Available
        partitioners include `tf.fixed_size_partitioner` and
        `tf.variable_axis_size_partitioner`.  For more details, see the
        documentation of `tf.get_variable` and the  "Variable Partitioners
        and Sharding" section of the API guide.

    Returns:
      The created variable.  Usually either a `Variable` or `ResourceVariable`
      instance.  If `partitioner` is not `None`, a `PartitionedVariable`
      instance is returned.

    Raises:
      RuntimeError: If called in Eager mode with regularizers.
    """
    if context.in_graph_mode():
      existing_variables = set(tf_variables.global_variables())
    if dtype is None:
      dtype = self.dtype or dtypes.float32

    self._set_scope(None)
    with vs.variable_scope(
        self._scope, reuse=(self.built or self._reuse)) as scope:
      with ops.name_scope(self._name_scope_name(scope)):
        variable = vs.get_variable(name,
                                   shape=shape,
                                   initializer=initializer,
                                   dtype=dtypes.as_dtype(dtype),
                                   constraint=constraint,
                                   trainable=trainable and self.trainable,
                                   partitioner=partitioner)
        if context.in_graph_mode():
          if (trainable and self.trainable
              and variable not in tf_variables.trainable_variables()):
            # A custom getter / variable scope overrode the trainable flag.
            trainable = False
          if variable in existing_variables:
            return variable
          if regularizer:
            # To match the behavior of tf.get_variable(), we only
            # apply regularization if the variable is newly created.
            if isinstance(variable, tf_variables.PartitionedVariable):
              for v in variable:
                with ops.colocate_with(v.op):
                  with ops.name_scope(name + '/Regularizer'):
                    regularization = regularizer(v)
                if regularization is not None:
                  self.add_loss(regularization)
            else:
              with ops.colocate_with(variable.op):
                with ops.name_scope(name + '/Regularizer'):
                  regularization = regularizer(variable)
              if regularization is not None:
                self.add_loss(regularization)
        elif regularizer:
          if isinstance(variable, tf_variables.PartitionedVariable):
            raise RuntimeError(
                'Partitioned variable regularization is not yet supported when '
                'executing eagerly. File a feature request is this is '
                'important to you.')
          # Save a zero-argument lambda which runs the regularizer on the
          # variable, to be executed when `Layer.losses` is requested. This
          # makes losses responsive to variable updates when executing eagerly.
          self._losses.append(lambda: regularizer(variable))
    if trainable:
      self._trainable_weights.append(variable)
    else:
      self._non_trainable_weights.append(variable)
    return variable

  def __call__(self, inputs, *args, **kwargs):
    """Wraps `call`, applying pre- and post-processing steps.

    Arguments:
      inputs: input tensor(s).
      *args: additional positional arguments to be passed to `self.call`.
      **kwargs: additional keyword arguments to be passed to `self.call`.
        **Note**: kwarg `scope` is reserved for use by the layer.

    Returns:
      Output tensor(s).

    Note:
      - If the layer's `call` method takes a `scope` keyword argument,
        this argument will be automatically set to the current variable scope.
      - If the layer's `call` method takes a `mask` argument (as some Keras
        layers do), its default value will be set to the mask generated
        for `inputs` by the previous layer (if `input` did come from
        a layer that generated a corresponding mask, i.e. if it came from
        a Keras layer with masking support.

    Raises:
      ValueError: if the layer's `call` method returns None (an invalid value).
    """
    self._set_scope(kwargs.pop('scope', None))
    input_list = nest.flatten(inputs)

    in_graph_mode = context.in_graph_mode()
    in_deferred_mode = isinstance(input_list[0], _DeferredTensor)
    # Ensure the Layer, if being reused, is working with inputs from
    # the same graph as where it was created.
    if in_graph_mode:
      try:
        ops._get_graph_from_inputs(input_list, graph=self.graph)  # pylint: disable=protected-access
      except ValueError as e:
        raise ValueError('Input graph and Layer graph are not the same: %s' % e)
    if in_graph_mode or in_deferred_mode:
      user_kwargs = copy.copy(kwargs)

    # Handle Keras mask propagation from previous layer to current layer.
    previous_mask = None
    if (not hasattr(self, '_compute_previous_mask') or
        self._compute_previous_mask):
      previous_mask = _collect_previous_mask(inputs)
      if ('mask' in estimator_util.fn_args(self.call) and
          'mask' not in kwargs and
          not _is_all_none(previous_mask)):
        # The previous layer generated a mask, and mask was not explicitly pass
        # to __call__, hence we set previous_mask as the default value.
        kwargs['mask'] = previous_mask

    if self.built:
      try:
        # Some classes which inherit from Layer do not use its constructor, so
        # rather than initializing to None we check for an AttributeError.
        scope_context_manager = self._always_reuse_variable_scope
      except AttributeError:
        # From this point we will always set reuse=True, so create a "final"
        # variable scope with this setting. We avoid re-creating variable scopes
        # after this point as an optimization.
        self._always_reuse_variable_scope = vs.variable_scope(
            self._scope, reuse=True)
        scope_context_manager = self._always_reuse_variable_scope
    else:
      scope_context_manager = vs.variable_scope(
          self._scope, reuse=self._reuse)
    with scope_context_manager as scope:
      with ops.name_scope(self._name_scope_name(scope)):
        if not self.built:
          if not in_graph_mode:
            # Activity regularization is currently unsupported in Eager mode.
            if self._activity_regularizer:
              raise ValueError('activity_regularizer currently unsupported in '
                               'Eager mode. Found an activity_regularizer in '
                               '%s(%s).' % (self.__class__.__name__, self))
          if not in_graph_mode and not in_deferred_mode:
            # TODO(agarwal): support _keras_history in Eager mode.
            for x in input_list:
              if hasattr(x, '_keras_history'):
                raise ValueError('_keras_history currently unsupported in '
                                 'Eager mode. Found _keras_history in %s while '
                                 'executing __call__ for %s(%s)' %
                                 (x, self.__class_.__name__, self))

          # Check input assumptions set before layer building, e.g. input rank.
          self._assert_input_compatibility(inputs)
          if input_list and self._dtype is None:
            try:
              self._dtype = input_list[0].dtype.name
            except AttributeError:
              pass
          input_shapes = nest.map_structure(lambda x: x.get_shape(), inputs)
          self.build(input_shapes)
        try:
          # Note: not all sub-classes of Layer call Layer.__init__ (especially
          # the ones under tensorflow/python/keras). Hence we recompute this
          # attribute here if it is not set.
          # TODO(agarwal): Fix the sub-classes and avoid this complexity.
          call_has_scope_arg = self._call_has_scope_arg
        except AttributeError:
          call_has_scope_arg = 'scope' in estimator_util.fn_args(self.call)
        if call_has_scope_arg:
          kwargs['scope'] = scope
        # Check input assumptions set after layer building, e.g. input shape.
        if in_graph_mode or in_deferred_mode:
          self._assert_input_compatibility(inputs)

        if not in_deferred_mode:
          outputs = self.call(inputs, *args, **kwargs)
          if outputs is None:
            raise ValueError('A layer\'s `call` method should return a Tensor '
                             'or a list of Tensors, not None.')
        else:
          # Deferred mode behavior: use `_compute_output_shape` to
          # infer the number of outputs of the layer and their shapes.
          output_shapes = self._compute_output_shape(input_shapes)
          output_shapes = nest.flatten(output_shapes)
          outputs = [
              # TODO(fchollet): name the deferred tensors?
              _DeferredTensor(shape=shape, dtype=self._dtype)
              for shape in output_shapes
          ]
          if len(outputs) == 1:
            outputs = outputs[0]

        if in_graph_mode:
          # Apply activity regularization.
          # Note that it should be applied every time the layer creates a new
          # output, since it is output-specific.
          if self._activity_regularizer:
            output_list = nest.flatten(outputs)
            for output in output_list:
              with ops.name_scope('ActivityRegularizer'):
                activity_regularization = self._activity_regularizer(output)
              self.add_loss(activity_regularization, inputs=inputs)

        if not in_deferred_mode:
          # TODO(fchollet): consider how masking will work with deferred mode.
          # Handle mask computation and propagation to the next layer.
          if hasattr(self, 'compute_mask'):
            output_mask = self.compute_mask(inputs, previous_mask)
            if isinstance(outputs, list):
              if output_mask is None:
                output_mask = [None for _ in range(len(outputs))]
              for x, m in zip(outputs, output_mask):
                x._keras_mask = m  # pylint: disable=protected-access
            else:
              outputs._keras_mask = output_mask  # pylint: disable=protected-access

    if in_graph_mode:
      # If all input tensors have history metadata,
      # we update the output tensors
      # with corresponding history metadata, thus eventually allowing to use
      # these tensors to instantiate a Network.
      if _have_all_keras_metadata(inputs):
        # If the layer returns tensors from its inputs, unmodified,
        # we copy them to avoid loss of tensor metadata.
        output_ls = nest.flatten(outputs)
        output_ls_copy = []
        for x in output_ls:
          if x in input_list:
            with ops.name_scope(scope.original_name_scope):
              x = array_ops.identity(x)
          output_ls_copy.append(x)
        if len(output_ls_copy) == 1:
          outputs = output_ls_copy[0]
        else:
          outputs = output_ls_copy

      # Update global default collections.
      _add_elements_to_collection(self.updates, ops.GraphKeys.UPDATE_OPS)

    if in_deferred_mode or in_graph_mode:
      if _have_all_keras_metadata(inputs):
        # Add an inbound node to the layer, so it can keep track of this call.
        # This updates the layer history of the output tensor(s).
        self._add_inbound_node(
            input_tensors=inputs, output_tensors=outputs, arguments=user_kwargs)

    self.built = True
    return outputs

  @property
  def dtype(self):
    return self._dtype

  @property
  def name(self):
    return self._name

  @property
  def activity_regularizer(self):
    """Optional regularizer function for the output of this layer."""
    return self._activity_regularizer

  @property
  def scope_name(self):
    if not self._scope:
      raise ValueError('No name available for layer scope because the layer "' +
                       self._name + '" has not been used yet. The scope name ' +
                       ' is determined the first time the layer instance is ' +
                       'called. You must therefore call the layer before ' +
                       'querying `scope_name`.')
    return self._scope.name

  @property
  def trainable_weights(self):
    return self._trainable_weights if self.trainable else []

  @property
  def non_trainable_weights(self):
    if self.trainable:
      return self._non_trainable_weights
    else:
      return self._trainable_weights + self._non_trainable_weights

  @property
  def trainable_variables(self):
    return self.trainable_weights

  @property
  def non_trainable_variables(self):
    return self.non_trainable_weights

  @property
  def weights(self):
    """Returns the list of all layer variables/weights.

    Returns:
      A list of variables.
    """
    return self.trainable_weights + self.non_trainable_weights

  @property
  def variables(self):
    """Returns the list of all layer variables/weights.

    Returns:
      A list of variables.
    """
    return self.weights

  @property
  def updates(self):
    if context.in_eager_mode():
      raise RuntimeError('Layer.updates not supported in Eager mode.')
    return self._updates

  def add_update(self, updates, inputs=None):
    """Add update op(s), potentially dependent on layer inputs.

    Weight updates (for instance, the updates of the moving mean and variance
    in a BatchNormalization layer) may be dependent on the inputs passed
    when calling a layer. Hence, when reusing the same layer on
    different inputs `a` and `b`, some entries in `layer.updates` may be
    dependent on `a` and some on `b`. This method automatically keeps track
    of dependencies.

    The `get_updates_for` method allows to retrieve the updates relevant to a
    specific set of inputs.

    This call is ignored in Eager mode.

    Arguments:
      updates: Update op, or list/tuple of update ops.
      inputs: Optional input tensor(s) that the update(s) depend on. Must
        match the `inputs` argument passed to the `__call__` method at the time
        the updates are created. If `None` is passed, the updates are assumed
        to be unconditional, and will apply across all dataflows of the layer.
    """
    if context.in_eager_mode():
      return  # Updates already applied when in eager mode.
    updates = _to_list(updates)
    if not updates:
      return
    self._updates += updates
    if inputs is not None:
      inputs = nest.flatten(inputs)
    if not inputs:
      inputs = None
    if inputs is not None:
      # We compute an ID that uniquely identifies the list of tensors.
      # This ID is order-sensitive.
      inputs_hash = layers_util.object_list_uid(inputs)
    else:
      inputs_hash = None
    if inputs_hash not in self._per_input_updates:
      self._per_input_updates[inputs_hash] = []
    self._per_input_updates[inputs_hash] += updates

  def get_updates_for(self, inputs):
    """Retrieves updates relevant to a specific set of inputs.

    Arguments:
      inputs: Input tensor or list/tuple of input tensors.
        Must match the `inputs` argument passed to the `__call__` method
        at the time the updates were created.
        If you pass `inputs=None`, unconditional updates are returned.

    Returns:
      List of update ops of the layer that depend on `inputs`.

    Raises:
      RuntimeError: If called in Eager mode.
    """
    if context.in_eager_mode():
      raise RuntimeError('Layer.get_updates_for not supported in Eager mode.')
    if inputs is not None:
      inputs = nest.flatten(inputs)
    if not inputs:
      inputs = None
    if inputs is not None:
      inputs_hash = layers_util.object_list_uid(inputs)
    else:
      inputs_hash = None
    return self._per_input_updates.get(inputs_hash, [])

  @property
  def losses(self):
    """Losses which are associated with this `Layer`.

    Note that when executing eagerly, getting this property evaluates
    regularizers. When using graph execution, variable regularization ops have
    already been created and are simply returned here.

    Returns:
      A list of tensors.
    """
    if context.in_eager_mode():
      # _losses may only contain variable regularization losses when executing
      # eagerly, and they have been saved as lambdas to be executed when
      # requested.
      return [regularizer() for regularizer in self._losses]
    else:
      return self._losses

  def add_loss(self, losses, inputs=None):
    """Add loss tensor(s), potentially dependent on layer inputs.

    Some losses (for instance, activity regularization losses) may be dependent
    on the inputs passed when calling a layer. Hence, when reusing the same
    layer on different inputs `a` and `b`, some entries in `layer.losses` may
    be dependent on `a` and some on `b`. This method automatically keeps track
    of dependencies.

    The `get_losses_for` method allows to retrieve the losses relevant to a
    specific set of inputs.

    Note that `add_loss` is not supported when executing eagerly. Instead,
    variable regularizers may be added through `add_variable`. Activity
    regularization is not supported directly (but such losses may be returned
    from `Layer.call()`).

    Arguments:
      losses: Loss tensor, or list/tuple of tensors.
      inputs: Optional input tensor(s) that the loss(es) depend on. Must
        match the `inputs` argument passed to the `__call__` method at the time
        the losses are created. If `None` is passed, the losses are assumed
        to be unconditional, and will apply across all dataflows of the layer
        (e.g. weight regularization losses).

    Raises:
      RuntimeError: If called in Eager mode.
    """
    if context.in_eager_mode():
      raise RuntimeError('Layer.add_loss not supported in Eager mode.')
    losses = _to_list(losses)
    if not losses:
      return
    self._losses += losses
    if inputs is not None:
      inputs = nest.flatten(inputs)
    if not inputs:
      inputs = None
    if inputs is not None:
      # We compute an ID that uniquely identifies the list of tensors.
      # This ID is order-sensitive.
      inputs_hash = layers_util.object_list_uid(inputs)
    else:
      inputs_hash = None
    if inputs_hash not in self._per_input_losses:
      self._per_input_losses[inputs_hash] = []
    self._per_input_losses[inputs_hash] += losses
    _add_elements_to_collection(losses, ops.GraphKeys.REGULARIZATION_LOSSES)

  def get_losses_for(self, inputs):
    """Retrieves losses relevant to a specific set of inputs.

    Arguments:
      inputs: Input tensor or list/tuple of input tensors.
        Must match the `inputs` argument passed to the `__call__`
        method at the time the losses were created.
        If you pass `inputs=None`, unconditional losses are returned,
        such as weight regularization losses.

    Returns:
      List of loss tensors of the layer that depend on `inputs`.

    Raises:
      RuntimeError: If called in Eager mode.
    """
    if context.in_eager_mode():
      raise RuntimeError('Layer.get_losses_for not supported in Eager mode.')
    if inputs is not None:
      inputs = nest.flatten(inputs)
    if not inputs:
      inputs = None
    if inputs is not None:
      inputs_hash = layers_util.object_list_uid(inputs)
    else:
      inputs_hash = None
    return self._per_input_losses.get(inputs_hash, [])

  def _name_scope_name(self, current_variable_scope):
    """Determines op naming for the Layer."""
    return current_variable_scope.original_name_scope

  def _compute_output_shape(self, input_shape):
    """Computes the output shape of the layer given the input shape.

    Assumes that the layer will be built to match that input shape.
    If this method is not implemented by child classes, the default
    assumption will be that the layer does not alter the shape of the tensors
    passing through it.

    Args:
      input_shape: A (possibly nested tuple of) `TensorShape`.  It need not
        be fully defined (e.g. the batch size may be unknown).

    Returns:
      A (possibly nested tuple of) `TensorShape`.

    Raises:
      TypeError: if `input_shape` is not a (possibly nested tuple of)
        `TensorShape`.
      ValueError: if `input_shape` is incomplete or is incompatible with the
        the layer.
    """
    return input_shape

  def _make_unique_name(self, name_uid_map=None, avoid_names=None,
                        namespace='', zero_based=False):
    base_name = _to_snake_case(self.__class__.__name__)
    name = _unique_layer_name(base_name, name_uid_map=name_uid_map,
                              avoid_names=avoid_names, namespace=namespace,
                              zero_based=zero_based)
    return (name, base_name)

  def _set_scope(self, scope=None):
    if self._scope is None:
      # If constructed with _scope=None, lazy setting of scope.
      if self._reuse:
        with vs.variable_scope(
            scope if scope is not None else self._base_name) as captured_scope:
          self._scope = captured_scope
      else:
        with vs.variable_scope(
            scope, default_name=self._base_name) as captured_scope:
          self._scope = captured_scope

  @property
  def graph(self):
    if context.in_eager_mode():
      raise RuntimeError('Layer.graph not supported in Eager mode.')
    return self._graph

  def __deepcopy__(self, memo):
    no_copy = set(['_graph'])
    shallow_copy = set(['_scope', '_always_reuse_variable_scope'])
    cls = self.__class__
    result = cls.__new__(cls)
    memo[id(self)] = result
    for k, v in self.__dict__.items():
      if k in no_copy:
        setattr(result, k, v)
      elif k in shallow_copy:
        setattr(result, k, copy.copy(v))
      elif _is_tensor_or_tensor_list(v):
        setattr(result, k, v)
      else:
        setattr(result, k, copy.deepcopy(v, memo))
    return result

  def apply(self, inputs, *args, **kwargs):
    """Apply the layer on a input.

    This simply wraps `self.__call__`.

    Arguments:
      inputs: Input tensor(s).
      *args: additional positional arguments to be passed to `self.call`.
      **kwargs: additional keyword arguments to be passed to `self.call`.

    Returns:
      Output tensor(s).
    """
    return self.__call__(inputs, *args, **kwargs)

  def _add_inbound_node(self,
                        input_tensors,
                        output_tensors,
                        arguments=None):
    """Internal method to create an inbound node for the layer.

    Arguments:
        input_tensors: list of input tensors.
        output_tensors: list of output tensors.
        arguments: dictionary of keyword arguments that were passed to the
            `call` method of the layer at the call that created the node.
    """
    input_tensors = nest.flatten(input_tensors)
    output_tensors = nest.flatten(output_tensors)

    # Collect input tensor(s) coordinates.
    inbound_layers = []
    node_indices = []
    tensor_indices = []
    for x in input_tensors:
      assert hasattr(x, '_keras_history')
      inbound_layer, node_index, tensor_index = x._keras_history  # pylint: disable=protected-access
      inbound_layers.append(inbound_layer)
      node_indices.append(node_index)
      tensor_indices.append(tensor_index)

    # Create node, add it to inbound nodes.
    Node(
        self,
        inbound_layers=inbound_layers,
        node_indices=node_indices,
        tensor_indices=tensor_indices,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        arguments=arguments)

    # Update tensor history metadata.
    for i in range(len(output_tensors)):
      # The metadata attribute consists of 1) a layer instance
      # 2) a node index for the layer, 3) a tensor index for the node.
      # The allows layer reuse (multiple nodes per layer) and multi-output
      # or multi-input layers (e.g. a layer can return multiple tensors,
      # and each can be sent to a different layer).
      output_tensors[i]._keras_history = (self, len(self._inbound_nodes) - 1, i)  # pylint: disable=protected-access

  def _get_node_attribute_at_index(self, node_index, attr, attr_name):
    """Private utility to retrieves an attribute (e.g. inputs) from a node.

    This is used to implement the methods:
        - get_input_shape_at
        - get_output_shape_at
        - get_input_at
        etc...

    Arguments:
        node_index: Integer index of the node from which
            to retrieve the attribute.
        attr: Exact node attribute name.
        attr_name: Human-readable attribute name, for error messages.

    Returns:
        The layer's attribute `attr` at the node of index `node_index`.

    Raises:
        RuntimeError: If the layer has no inbound nodes, or if called in Eager
        mode.
        ValueError: If the index provided does not match any node.
    """
    assert context.in_graph_mode()
    if not self._inbound_nodes:
      raise RuntimeError('The layer has never been called '
                         'and thus has no defined ' + attr_name + '.')
    if not len(self._inbound_nodes) > node_index:
      raise ValueError('Asked to get ' + attr_name + ' at node ' +
                       str(node_index) + ', but the layer has only ' +
                       str(len(self._inbound_nodes)) + ' inbound nodes.')
    values = getattr(self._inbound_nodes[node_index], attr)
    if len(values) == 1:
      return values[0]
    else:
      return values

  def get_input_shape_at(self, node_index):
    """Retrieves the input shape(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A shape tuple
        (or list of shape tuples if the layer has multiple inputs).

    Raises:
      RuntimeError: If called in Eager mode.
    """
    if context.in_eager_mode():
      raise RuntimeError(
          'Layer.get_input_shape_at not supported in Eager mode.')
    return self._get_node_attribute_at_index(node_index, 'input_shapes',
                                             'input shape')

  def get_output_shape_at(self, node_index):
    """Retrieves the output shape(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A shape tuple
        (or list of shape tuples if the layer has multiple outputs).

    Raises:
      RuntimeError: If called in Eager mode.
    """
    if context.in_eager_mode():
      raise RuntimeError(
          'Layer.get_output_shape_at not supported in Eager mode.')
    return self._get_node_attribute_at_index(node_index, 'output_shapes',
                                             'output shape')

  def get_input_at(self, node_index):
    """Retrieves the input tensor(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A tensor (or list of tensors if the layer has multiple inputs).

    Raises:
      RuntimeError: If called in Eager mode.
    """
    if context.in_eager_mode():
      raise RuntimeError('Layer.get_input_at not supported in Eager mode.')
    return self._get_node_attribute_at_index(node_index, 'input_tensors',
                                             'input')

  def get_output_at(self, node_index):
    """Retrieves the output tensor(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A tensor (or list of tensors if the layer has multiple outputs).

    Raises:
      RuntimeError: If called in Eager mode.
    """
    if context.in_eager_mode():
      raise RuntimeError('Layer.get_output_at not supported in Eager mode.')
    return self._get_node_attribute_at_index(node_index, 'output_tensors',
                                             'output')

  @property
  def input(self):
    """Retrieves the input tensor(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer.

    Returns:
        Input tensor or list of input tensors.

    Raises:
        AttributeError: if the layer is connected to
        more than one incoming layers.

    Raises:
      RuntimeError: If called in Eager mode.
      AttributeError: If no inbound nodes are found.
    """
    if context.in_eager_mode():
      raise RuntimeError('Layer.input not supported in Eager mode.')
    if not self._inbound_nodes:
      raise AttributeError('Layer ' + self.name +
                           ' is not connected, no input to return.')
    return self._get_node_attribute_at_index(0, 'input_tensors', 'input')

  @property
  def output(self):
    """Retrieves the output tensor(s) of a layer.

    Only applicable if the layer has exactly one output,
    i.e. if it is connected to one incoming layer.

    Returns:
      Output tensor or list of output tensors.

    Raises:
      AttributeError: if the layer is connected to more than one incoming
        layers.
      RuntimeError: if called in Eager mode.
    """
    if context.in_eager_mode():
      raise RuntimeError('Layer.output not supported in Eager mode.')
    if not self._inbound_nodes:
      raise AttributeError('Layer ' + self.name + ' has no inbound nodes.')
    return self._get_node_attribute_at_index(0, 'output_tensors', 'output')

  @property
  def input_shape(self):
    """Retrieves the input shape(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer, or if all inputs
    have the same shape.

    Returns:
        Input shape, as an integer shape tuple
        (or list of shape tuples, one tuple per input tensor).

    Raises:
        AttributeError: if the layer has no defined input_shape.
        RuntimeError: if called in Eager mode.
    """
    if context.in_eager_mode():
      raise RuntimeError('Layer.input_shape not supported in Eager mode.')
    if not self._inbound_nodes:
      raise AttributeError('The layer has never been called '
                           'and thus has no defined input shape.')
    all_input_shapes = set(
        [str(node.input_shapes) for node in self._inbound_nodes])
    if len(all_input_shapes) == 1:
      input_shapes = self._inbound_nodes[0].input_shapes
      if len(input_shapes) == 1:
        return tuple(tensor_shape.TensorShape(input_shapes[0]).as_list())
      else:
        return [
            tuple(tensor_shape.TensorShape(shape).as_list())
            for shape in input_shapes
        ]
    else:
      raise AttributeError('The layer "' + str(self.name) +
                           ' has multiple inbound nodes, '
                           'with different input shapes. Hence '
                           'the notion of "input shape" is '
                           'ill-defined for the layer. '
                           'Use `get_input_shape_at(node_index)` '
                           'instead.')

  def count_params(self):
    """Count the total number of scalars composing the weights.

    Returns:
        An integer count.

    Raises:
        ValueError: if the layer isn't yet built
          (in which case its weights aren't yet defined).
    """
    if not self.built:
      if self.__class__.__name__ == 'Sequential':
        self.build()  # pylint: disable=no-value-for-parameter
      else:
        raise ValueError('You tried to call `count_params` on ' + self.name +
                         ', but the layer isn\'t built. '
                         'You can build it manually via: `' + self.name +
                         '.build(batch_input_shape)`.')
    weight_shapes = [w.get_shape().as_list() for w in self.weights]
    return int(sum([np.prod(w) for w in weight_shapes]))

  @property
  def output_shape(self):
    """Retrieves the output shape(s) of a layer.

    Only applicable if the layer has one output,
    or if all outputs have the same shape.

    Returns:
        Output shape, as an integer shape tuple
        (or list of shape tuples, one tuple per output tensor).

    Raises:
        AttributeError: if the layer has no defined output shape.
        RuntimeError: if called in Eager mode.
    """
    if context.in_eager_mode():
      raise RuntimeError('Layer.output_shape not supported in Eager mode.')
    if not self._inbound_nodes:
      raise AttributeError('The layer has never been called '
                           'and thus has no defined output shape.')
    all_output_shapes = set(
        [str(node.output_shapes) for node in self._inbound_nodes])
    if len(all_output_shapes) == 1:
      output_shapes = self._inbound_nodes[0].output_shapes
      if len(output_shapes) == 1:
        return tuple(tensor_shape.TensorShape(output_shapes[0]).as_list())
      else:
        return [
            tuple(tensor_shape.TensorShape(shape).as_list())
            for shape in output_shapes
        ]
    else:
      raise AttributeError('The layer "%s"'
                           ' has multiple inbound nodes, '
                           'with different output shapes. Hence '
                           'the notion of "output shape" is '
                           'ill-defined for the layer. '
                           'Use `get_output_shape_at(node_index)` '
                           'instead.' % self.name)

  @property
  def inbound_nodes(self):
    """Deprecated, do NOT use! Only for compatibility with external Keras."""
    return self._inbound_nodes

  @property
  def outbound_nodes(self):
    """Deprecated, do NOT use! Only for compatibility with external Keras."""
    return self._outbound_nodes

  def _assert_input_compatibility(self, inputs):
    """Checks compatibility between the layer and provided inputs.

    This checks that the tensor(s) `inputs` verify the input assumptions
    of the layer (if any). If not, a clear and actional exception gets raised.

    Arguments:
        inputs: input tensor or list of input tensors.

    Raises:
        ValueError: in case of mismatch between
            the provided inputs and the expectations of the layer.
    """
    if not self.input_spec:
      return
    if not isinstance(self.input_spec, (list, tuple)):
      input_spec = nest.flatten(self.input_spec)
    else:
      input_spec = self.input_spec
    inputs = nest.flatten(inputs)
    if len(inputs) != len(input_spec):
      raise ValueError('Layer ' + self.name + ' expects ' +
                       str(len(input_spec)) + ' inputs, '
                       'but it received ' + str(len(inputs)) +
                       ' input tensors. Inputs received: ' + str(inputs))
    for input_index, (x, spec) in enumerate(zip(inputs, input_spec)):
      if spec is None:
        continue

      if (spec.ndim is not None or
          spec.min_ndim is not None or
          spec.max_ndim is not None):
        if x.get_shape().ndims is None:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           'its rank is undefined, but the layer requires a '
                           'defined rank.')

      # Check ndim.
      if spec.ndim is not None:
        ndim = x.get_shape().ndims
        if ndim != spec.ndim:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           'expected ndim=' + str(spec.ndim) + ', found ndim=' +
                           str(ndim) + '. Full shape received: ' +
                           str(x.get_shape().as_list()))
      if spec.max_ndim is not None:
        ndim = x.get_shape().ndims
        if ndim is not None and ndim > spec.max_ndim:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           'expected max_ndim=' + str(spec.max_ndim) +
                           ', found ndim=' + str(ndim))
      if spec.min_ndim is not None:
        ndim = x.get_shape().ndims
        if ndim is not None and ndim < spec.min_ndim:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           ': expected min_ndim=' + str(spec.min_ndim) +
                           ', found ndim=' + str(ndim) +
                           '. Full shape received: ' +
                           str(x.get_shape().as_list()))
      # Check dtype.
      if spec.dtype is not None:
        if x.dtype != spec.dtype:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           'expected dtype=' + str(spec.dtype) +
                           ', found dtype=' + str(x.dtype))
      # Check specific shape axes.
      if spec.axes:
        shape = x.get_shape().as_list()
        if shape is not None:
          for axis, value in spec.axes.items():
            if hasattr(value, 'value'):
              value = value.value
            if value is not None and shape[int(axis)] not in {value, None}:
              raise ValueError(
                  'Input ' + str(input_index) + ' of layer ' + self.name + ' is'
                  ' incompatible with the layer: expected axis ' + str(axis) +
                  ' of input shape to have value ' + str(value) +
                  ' but received input with shape ' + str(shape))
      # Check shape.
      if spec.shape is not None:
        shape = x.get_shape().as_list()
        if shape is not None:
          for spec_dim, dim in zip(spec.shape, shape):
            if spec_dim is not None and dim is not None:
              if spec_dim != dim:
                raise ValueError('Input ' + str(input_index) +
                                 ' is incompatible with layer ' + self.name +
                                 ': expected shape=' + str(spec.shape) +
                                 ', found shape=' + str(shape))
```

