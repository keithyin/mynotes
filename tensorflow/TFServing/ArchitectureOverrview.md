# 结构一览（Architecture Overview）
`Tensorflow Serving` 是为了生产环境设计的。他是 机器学习模型 的一个灵活，高性能的服务系统。 `TensorFlow Serving` 使得部署一个新算法或者实验非常简单，它保持了同样高的服务器架构和`APIs`。 `TensorFlow Serving` 提供了一个与`TensorFlow`模型的一个集成, 但是它也可以扩展到其他模型上去。

## 关键概念 (Key Concepts)

为了理解`TensorFlow Serving`的架构，必须要理解以下几个关键概念：

**Servables**

`Servables` 是`TensorFlow Serving` 中的一个 重要抽象。 `Servables` 是 `client` 用来执行计算（例如，查找和推断）的基本对象。

`Servable`的大小和粒度是非常灵活的。 单一的 `Servable` 可能包含  从`lookup table`的一个碎片，到一个模型，再一组推理模型的任何东西。`Servables` 可以是任何类型和接口，使能够具有灵活性和将来的提升，例如:

* 流型结果(streaming result)
* 实验API
* `operation` 的异步模式

`Servables` 不管理它们自己的生存周期。

典型的`servables` 包含以下东西：

* a TensorFlow SavedModelBundle (tensorflow::Session)
* a lookup table for embedding or vocabulary lookups


**Servable 版本**


`TensorFlow Serving` 在一个服务实例中，可以处理一个或多个版本的`servale`。这使得新鲜的算法配置，权重，和其它数据可以随时被夹在。  `Versions` 使得一个`servable`的多个版本可以同时被加载，支持逐步展开和实验。在服务期间，`clients` 可以请求最新的版本，也可以通过指定版本`id`来请求指定的模型。

**Servable Streams**

一个 `servable stream` 是 `servable` 的版本序列，按照版本序号排序。

**Models**

`TensorFlow Serving` 将一个`model`表示为一个或多个 `servables`。 一个机器学习的模型 可能会包含一个或多个算法(包括学习到的权重) 和  `lookup` 或 `embedding` 表。

我们可以把一个混合模型使用以下几种方式来表示：

* multiple independent servables
* single composite servable

一个 `servable` 也可以对应模型的一部分。例如，一个大的`lookup table` 可能被切片到多个 `TensorFlow Serving` 实例上。

**Loaders**

`Loaders` 管理 `servable` 的生命周期。  `Loader API` 实现了一个通用的架构，它独立于 学习算法，数据或产品用例。准确的说，`Loaders`  标准化了 加载和卸载 `servable` 的 `API`。

**Sources**

`Sources` 是 产生`servable` 的 插件模块;每个 `Source` 产生0个或多个 `servable streams`。对于每个`stream`来说，`Source`为每个版本提供一个 `Loader`实例 去加载它。 (准确的说,一个 `Source` 是由0个或多个`SourceAdapters`连接起来的，并且链中的最后一个负责释放 `Loader`。)

`TensorFlow Serving` 为 `Sources` 提供的接口简单又狭隘，因此，它使你可以使用任意的存储系统去发现`servables`，然后加载它。`Sources` 可能需要访问其它机制，例如`RPC`。`TensorFlow Serving` 包含公共参考源实现， 例如： `TensorFlow Serving` 可以轮询文件系统。


`Sources` 可以容纳多个`servables` 或版本  共享的状态，特殊情况下，例如有效接受增量更新的模型


**Aspired Versions**

Aspired versions represent the set of servable versions that should be loaded and ready. Sources communicate this set of servable versions for a single servable stream at a time. When a Source gives a new list of aspired versions to the Manager, it supercedes the previous list for that servable stream. The Manager unloads any previously loaded versions that no longer appear in the list.

See the advanced tutorial to see how version loading works in practice.

**Managers**

Managers handle the full lifecycle of Servables, including:

loading Servables
serving Servables
unloading Servables
Managers listen to Sources and track all versions. The Manager tries to fulfill Sources’ requests, but may refuse to load an aspired version if, say, required resources aren’t available. Managers may also postpone an “unload”. For example, a Manager may wait to unload until a newer version finishes loading, based on a policy to guarantee that at least one version is loaded at all times.

TensorFlow Serving Managers provide a simple, narrow interface – GetServableHandle() – for clients to access loaded servable instances.

**Core**

TensorFlow Serving Core manages (via standard TensorFlow Serving APIs) the following aspects of servables:

lifecycle
metrics
TensorFlow Serving Core treats servables and loaders as opaque objects.

## Life of a Servable

tf serving architecture diagram

Broadly speaking:

Sources create Loaders for Servable Versions.
Loaders are sent as Aspired Versions to the Manager, which loads and serves them to client requests.
In more detail:

A Source plugin creates a Loader for a specific version. The Loader contains whatever metadata it needs to load the Servable.
The Source uses a callback to notify the Manager of the Aspired Version.
The Manager applies the configured Version Policy to determine the next action to take, which could be to unload a previously loaded version or to load the new version.
If the Manager determines that it’s safe, it gives the Loader the required resources and tells the Loader to load the new version.
Clients ask the Manager for the Servable, either specifying a version explicitly or just requesting the latest version. The Manager returns a handle for the Servable.
For example, say a Source represents a TensorFlow graph with frequently updated model weights. The weights are stored in a file on disk.

The Source detects a new version of the model weights. It creates a Loader that contains a pointer to the model data on disk.
The Source notifies the Dynamic Manager of the Aspired Version.
The Dynamic Manager applies the Version Policy and decides to load the new version.
The Dynamic Manager tells the Loader that there is enough memory. The Loader instantiates the TensorFlow graph with the new weights.
A client requests a handle to the latest version of the model, and the Dynamic Manager returns a handle to the new version of the Servable.
## Extensibility

TensorFlow Serving provides several extension points where you can add new functionality.

Version Policy

Version Policies specify the sequence of version loading and unloading within a single servable stream.

TensorFlow Serving includes two policies that accommodate most known use- cases. These are the Availability Preserving Policy (avoid leaving zero versions loaded; typically load a new version before unloading an old one), and the Resource Preserving Policy (avoid having two versions loaded simultaneously, thus requiring double the resources; unload an old version before loading a new one). For simple usage of TensorFlow Serving where the serving availability of a model is important and the resource costs low, the Availability Preserving Policy will ensure that the new version is loaded and ready before unloading the old one. For sophisticated usage of TensorFlow Serving, for example managing versions across multiple server instances, the Resource Preserving Policy requires the least resources (no extra buffer for loading new versions).

Source

New Sources could support new filesystems, cloud offerings and algorithm backends. TensorFlow Serving provides some common building blocks to make it easy & fast to create new sources. For example, TensorFlow Serving includes a utility to wrap polling behavior around a simple source. Sources are closely related to Loaders for specific algorithms and data hosting servables.

See the Custom Source document for more about how to create a custom Source.

Loaders

Loaders are the extension point for adding algorithm and data backends. TensorFlow is one such algorithm backend. For example, you would implement a new Loader in order to load, provide access to, and unload an instance of a new type of servable machine learning model. We anticipate creating Loaders for lookup tables and additional algorithms.

See the Custom Servable document to learn how to create a custom servable.

Batcher

Batching of multiple requests into a single request can significantly reduce the cost of performing inference, especially in the presence of hardware accelerators such as GPUs. TensorFlow Serving includes a request batching widget that lets clients easily batch their type-specific inferences across requests into batch requests that algorithm systems can more efficiently process. See the Batching Guide for more information.

Next Steps

To get started with TensorFlow Serving, try the Basic Tutorial.
