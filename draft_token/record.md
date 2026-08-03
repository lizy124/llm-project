# vLLM 并行实现

软件版本：本文内容基于 vLLM 0.25.0 release。

硬件环境：除非特别声明，本文讨论的硬件及性能表现均基于 NVIDIA GPU 系列。

文章定位：本文非源码走读指南。文章侧重于架构与机制探讨，必要时提供部分代码细节。如需代码层面的细节，请查阅官方源码。

目标读者：本文适合有一定大模型推理基础，并且了解各种并行概念以及特点的开发者阅读。

在大模型（LLM）推理优化领域，并行计算技术通过在不同维度上解耦和拆分推理过程，提升了系统吞吐量，降低了延迟。目前业界主流的并行策略包括张量并行（TP，Tensor Parallelism）、流水线并行（PP，Pipeline Parallelism）、上下文并行（CP，Context Parallelism）、数据并行（DP，Data Parallelism）以及专家并行（EP，Expert Parallelism）等。

本文旨在从机制层面探讨这些并行手段在 vLLM 框架中的落地实现。为保持行文聚焦，本文暂时剥离 KV Transfer、EP、无状态模型、模型权重加载及底层通信算子的实现细节，将核心实现聚焦于 TP、PP、CP 等并行组的管理与构建。

本文内容将围绕以下几部分展开：

- 硬件抽象结构：理解 vLLM 对物理硬件的逻辑建模。
- 并行配置解析：追踪 vLLM 对用户并行意图的解析链路。
- 并行实现：深入剖析 vLLM 支撑各种并行策略的实现，将 `Executor -> Worker -> Device` 的过程分为两大部分进行探讨，同时穿插重要组件介绍。

## 1. 硬件抽象结构

如图所示，vLLM 通过分层架构将业务逻辑与底层复杂硬件拓扑解耦。为便于后续对并行机制的探讨，我们先对各层抽象做简要对齐：

- Client 层（接入网关）：专注外部请求的收发，与底层计算逻辑物理隔离。
- Core 层（调度逻辑）：持有 Scheduler 和 KVCacheManager，负责实现 Continuous Batching 和 PagedAttention。本层专职于请求级和 Token 级的逻辑调度，属于硬件无关层。
- Executor 层（执行引擎）：核心隔离层。向上为 Core 提供统一 API，避免底层异构硬件信息污染核心调度逻辑；向下负责解析并管理跨卡、跨节点的计算调度方案。
- Worker 层（计算实体）：对 Device（GPU）的直接建模。负责拉起计算资源、初始化分布式通信后端，并在指定硬件上执行实际计算流。
- ModelRunner 层（模型抽象）：在 Worker 内部屏蔽各类 LLM 的网络结构差异。

vLLM 并行相关的核心逻辑，如多进程拉起、通信组划分、设备映射等，主要收敛在由 Executor 到 Worker 的调度链路中。因此，本文后续讨论聚焦于 Executor 层与 Worker 层，暂时略过 Client、Core 与 ModelRunner 层。

## 2. 并行配置解析

vLLM 使用 `arg_utils.py` 中的 `create_engine_config()` 构造 `VllmConfig` 作为推理过程中的全局配置。在 `VllmConfig` 中，`ParallelConfig` 即为并行相关配置。

并行相关配置的核心主要有以下几点：

- 基础切分维度：`tensor_parallel_size`、`pipeline_parallel_size`、`prefill_context_parallel_size`、`data_parallel_size` 等。
- 全局派生维度：`world_size` 和 `world_size_across_dp` 等。一些由 `__post_init__()` 根据其他配置计算得来，另一些通过 Python 的 `@property` 机制计算得来。
- MoE 与专家并行：`enable_expert_parallel`、`enable_eplb` 和 `all2all_backend` 等。
- 运行时后端与拓扑：`distributed_executor_backend`、`numa_bind`、`numa_bind_cpus` 等。

`ParallelConfig` 中还使用了 `_validate_parallel_config()` 方法进行防御性校验。

## 3. 并行实现

在理清硬件抽象结构与配置解析逻辑后，本文将深入探讨 vLLM 并行策略的落地实现。vLLM 通过前文提到的 `distributed_executor_backend` 参数提供了对多种分布式后端的支持，例如 Ray 和原生多进程机制。

为了聚焦底层核心逻辑，剥离外部框架带来的复杂性，本文将以原生多进程 `MultiprocExecutor` 为后端探讨落地实现过程。

从宏观上看，整个并行策略实现的最终目标为：严格依据配置中预设的硬件拓扑结构，精准构建分布式进程组（Process Group），并为各个维度初始化底层通信机制。

我们将整个并行组的构建与通信过程拆解为两个核心阶段：

- 控制面的调度与下发（从 Executor 到 Worker）：主进程如何拉起子进程并分配物理计算资源。
- 计算节点内部的组网（从 Worker 到 GroupCoordinator）：各个物理节点如何建立通信域并完成分布式协调。

### 3.1 控制面的调度与下发

从 Executor 到 Worker，代码逻辑和业务逻辑的差别很大。仅 Executor 初始化就要经历 `Executor -> WorkerProc.make_worker_process -> Worker.worker_main -> WorkerProc.__init__ -> WorkerWrapperBase.init_worker -> ...` 等多层跳转。

为了避免变成代码走读，本文将控制面的调度与下发整理为逻辑上的四个切面，帮助读者后续自行查看源码：

- 拓扑与角色：解决怎么分工。
- 资源隔离与硬件亲和性：解决怎么榨干硬件性能。
- 进程间通信：解决主进程与子进程如何通信。
- 生命周期维护：解决兜底与排错。

不只是 vLLM，其他框架在落地并行实现时通常也需要围绕这四个切面展开。

#### 3.1.1 拓扑与角色

vLLM 通过以下 `ParallelConfig` 属性确定集群拓扑。

硬件资源：

- `nnodes`：集群里总共有几台物理机。
- `node_rank`：当前代码跑在哪一台物理机上。

基础切分维度：

- `tensor_parallel_size`：合作计算同一层神经网络的 GPU 数量。
- `prefill_context_parallel_size`：共同处理同一个超长请求的 GPU 数量。
- `pipeline_parallel_size`：组成一条完整前向计算的 GPU 或 GPU 组数量。
- `data_parallel_size`：全局范围内拥有完整模型推理能力的副本总数。
- `data_parallel_size_local`：单台物理机上运行的完整模型副本数量，至少为 1。该参数涉及多机、单机、内部负载均衡、外部负载均衡等多种组合情况，详情可见 `arg_utils.py`。

全局派生维度：

- `world_size`：运行一个完整模型副本需要多少 GPU，`world_size = tensor_parallel_size * prefill_context_parallel_size * pipeline_parallel_size`。
- `data_parallel_node_size`：全局 DP 包含多少个 Node，`data_parallel_node_size = data_parallel_size // data_parallel_size_local`。
- `nnodes_within_dp`：单个 DP 组内包含多少个 Node，`nnodes_within_dp = nnodes // data_parallel_node_size`。
- `local_world_size`：单台物理机有多少 GPU 参与了一个模型副本，`local_world_size = world_size // nnodes_within_dp`。

单台物理机或 GPU 的 Rank 计算：

- `node_rank_within_dp`：在一个 DP 组内，单台物理机的 Rank。
- `global_start_rank`：这里的 `global` 不是真正的全局 `nnodes` 维度，而是 `world_size` 维度，也就是一个 Node 里面的范围。因此 `global_start_rank = local_world_size * node_rank_within_dp`。
- `local_rank`：范围为 `[0, local_world_size)`。
- `global_rank`：`global_rank = global_start_rank + local_rank`。

由上可知，拓扑相关变量十分复杂，尤其是和 DP 相关的变量容易绕来绕去，命名也容易产生歧义。下面用一个例子说明。

例子中的硬件信息和基础并行维度如下：

- `nnodes`：4 台物理机。
- `node_rank`：各机器为 `[0, 1, 2, 3]`。
- `tensor_parallel_size`：8，一台单机上 8 个 GPU 全部参与 TP 并行组。
- `prefill_context_parallel_size`：假设为 1，减少画图复杂度。
- `pipeline_parallel_size`：2，两台物理机一起组成一个 PP 并行组。

全局派生维度如下：

- `data_parallel_size_local`：例子中单台机器没有办法装下一整个模型副本，需要两台机器组合流水线并行组才能完整装下一个副本，所以 `data_parallel_size_local` 为 1。也可以理解为这台机器参与多少个 DP 并行 Rank。
- `data_parallel_node_size`：全局有多少个 Node 参与并行，即两个大虚线方框。
- `nnodes_within_dp`：一个 Node 里面有多少台机器。因为 `nnodes = 4` 且 `data_parallel_node_size = 2`，所以每个 Node 有 `4 // 2 = 2` 台单机。
- `world_size`：按公式计算即可，不过此例中 `world_size` 跨了两台物理机。
- `local_world_size`：每台物理机中参与一个模型副本的 GPU 总数，用一个模型副本所需 GPU 总数除以一个 Node 内的物理机数量，即 `16 // 2 = 8`。

Rank 计算如下：

- `node_rank_within_dp`：一个 Node 里面两台机器，为 `[0, 1]`。
- `global_start_rank`：`world_size` 为 16，跨了两台机器的 16 个 GPU，所以两台单机的 `global_start_rank` 为 `[0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8]`。
- `local_rank`：`local_world_size` 为 8，因此每台单机的 `local_rank` 为 `[0, 1, 2, 3, 4, 5, 6, 7]`。
- `global_rank`：`global_start_rank + local_rank`，结果为 `[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]`。

拓扑信息完成构建后，还要注意一些关键 Rank 的 GPU 或物理机角色，它们需要承担更多职责：

- `node_rank_within_dp = 0`：该物理机为主节点，主节点中的主进程会创建全局广播消息队列，用来进行进程间、跨机器进程间通信。
- `global_rank % tensor_parallel_size == 0`：各 TP 组的首节点，要承担数据汇聚和调度职责。

#### 3.1.2 资源隔离与硬件亲和性

这部分主要在 `set_multiprocessing_worker_envs()` 方法中。

##### 3.1.2.1 资源隔离

进程的创建方法有 `fork` 和 `spawn` 之分。

`fork` 创建的子进程会直接复制主进程的内存空间等资源，效率较高，但共享内存空间会造成一些致命问题。例如 CUDA Context 一旦建立，主进程和子进程共用时可能发生 segmentation fault 或死锁。

`spawn` 会重新创建一整套资源，虽然效率较低，但不会出现上述资源共享问题。因此，为了兼顾效率与资源隔离，vLLM 需要对使用 `fork` 还是 `spawn` 进行选择。判断方法可见附录 4.1。

同时，PyTorch 底层使用 OpenMP 加速 CPU 上的计算，例如数据预处理和张量拼接。默认情况下，PyTorch 会为每个进程分配与机器物理核心数相等的线程来最大化加速。假设机器是 64 核 CPU、8 张 GPU，就会有 8 个进程，每个进程 64 个线程，总计 512 个线程，系统运行时会发生频繁上下文切换，严重占用 CPU 资源。

在非 CPU 后端的推理环境中，需要限制每个进程只分配一个 CPU 线程，用于对 GPU 侧进行控制。

##### 3.1.2.2 NUMA 绑核与网卡绑定

NUMA（Non-Uniform Memory Access，非统一内存访问）相关信息见附录 4.2。具体的 GPU、CPU 和 NUMA Node 的关联计算按照拓扑结构计算 Rank 即可，这里源码更加清晰。

此处需要关注的是 vLLM 如何将命令行参数放进子进程创建过程：

1. 构建不同的 `numactl` 命令行参数。EngineCore 主进程是一对多，Worker 子进程是一对一。
2. 检测系统是否安装了 `numactl` 命令，同时获取项目中的 `numa_wrapper.sh` 脚本路径。
3. 将命令行参数存入特定环境变量。
4. 将默认 Python 解释器替换为包装脚本，因此创建进程时默认的 Python 命令会被包装脚本接管。
5. 此过程全程通过 `yield` 包裹，在完成所有进程创建操作后，执行 `finally` 中的代码，将执行环境全部还原。

GPU 需要绑定最近的网卡来实现跨节点传输的最大效率。方式是根据 GPU 逻辑索引定位到 GPU 真实物理地址，然后找到最近的网卡地址进行绑定，并更新对应环境变量。这部分可直接查看源码。

#### 3.1.3 进程间通信

本节讨论 Executor 和 Worker 之间的通信。一般来说，通信可以分为两个部分：

- 控制面：用来完成进程的创建、同步与监测。
- 数据面：用来在推理过程中完成各进程间的信息同步。

关于数据面，由于使用到了 MessageQueue，本文会在 3.2 节讨论。

vLLM 使用普通管道（Pipe）完成 Executor 与 Worker 控制面上的通信：

- Executor 和每一个 Worker 均通过两个管道进行通信。
- 一个是 Worker 用来通知 Executor 初始化已经完成的 `ready` 管道。
- 另一个是 Executor 用来通知 Worker 执行资源销毁的 `death` 管道。

所有管道的 `duplex` 都置为 `False`，即单向传输。因为 Executor 只需要收 `ready` 管道的信息、发 `death` 管道的信息，而 Worker 只需要发 `ready` 管道的信息、收 `death` 管道的信息。

同时，由于管道和文件描述符的机制（见附录 4.3），各进程应该 `close()` 自己不需要的描述符，避免出现僵尸进程问题。

对于 `fork`，情况会更加特殊。vLLM 使用循环对各个 Rank 的 Worker 进程进行创建，后续创建的 Worker 会继承创建前面 Worker 时的文件描述符。因此，需要使用一个 `inherited_fds` 整型列表，不断追加前面 Worker 的文件描述符，并在当前 Worker 创建过程中将这些描述符关闭。

#### 3.1.4 生命周期维护

Worker 进程会专门调用 `monitor_death_pipe()`，启动一个线程对 `death_pipe` 进行监听，直到 Executor 发送信息到 `death_pipe` 中，随后执行 `shutdown()`。

vLLM 在创建 Worker 进程时，还通过 `signal.signal()` 对操作系统级别的硬中断（Signal）进行拦截。

`signal_handler` 是拦截到信号时的操作。`SIGTERM` 由操作系统、其他进程或系统管理员通过 `kill` 命令发送，而 `SIGINT` 由用户在终端输入中断字符触发，通常是 `Ctrl + C`。

vLLM 通过这种方式将操作系统级别的硬中断转化为 Python 层面的异常，从而正确销毁进程，防止 GPU 显存泄露、通信死锁等暴力停机可能导致的问题。

同时，vLLM 使用 `threading.Event()` 的 `is_set()` 方法保证用户多次按 `Ctrl + C` 时只抛出一次异常，避免重复操作。

为了保证 Executor 和 Worker 通信无误，vLLM 还设计了严格的同步策略：必须在所有 Worker 返回 ready 信息并且通信方式全部就位后，才展开后续工作。

最后，为了方便定位问题，vLLM 使用 `setup_proc_title_and_log_prefix()` 为每个进程起名字，在日志中可以明显看到是哪个 Rank 的进程出了问题。

### 3.2 MessageQueue

#### 3.2.1 Executor 与 Worker 通信的难题

我们不希望控制指令成为大模型推理时延中的瓶颈，因此 Executor 和 Worker 进程间通信的效率非常重要。

Python 自带的进程间通信本质上是点对点通信模型，天然不支持广播。而 Executor 和所有 Worker 必须保持信息同步。如果使用原生 IPC，就只能写循环，给每个 Worker 进程发送一遍。因此，需要引入发布-订阅或一对多广播的通信模型，也就是消息队列。

同时，为了最大化提高效率，希望能够使用共享内存传递消息，避免像 Socket 一样拷贝到缓冲区再发送。

不过使用共享内存的消息队列会面临状态同步问题。Executor 必须确保队列中的消息已经被所有 Worker 接收完毕，才能进行下一轮覆盖写入。为了解决同步问题，又必须引入锁。但锁会明显影响整体前向过程的推理性能，所以必须使用无锁算法控制同步。

另外，如果模型过大，导致 Executor 和 Worker 不在同一台物理机上，共享内存就失效了，必须退回到跨机 RPC 调用。

综上，需要一个远程时使用 Socket 通信、本地时使用共享内存通信、并且支持无锁同步的消息队列。vLLM 的答案是自己实现一个 MessageQueue。

本文先介绍 MessageQueue 的重要组件，然后将所有组件串联形成完整广播过程。

#### 3.2.2 ShmRingBuffer

`ShmRingBuffer` 是 MessageQueue 中用来实现共享内存和无锁自旋的组件。源码注释提供了 `ShmRingBuffer` 的图形介绍，这里从注释中抽象出重点。

图中上半部分的方块代表 `ShmRingBuffer.shared_memory`，作为 `ShmRingBuffer` 存储数据的容器，类型为 Python 内置的 `SharedMemory`。从 metadata 那一侧可以看出，`ShmRingBuffer` 仅针对 `1 writer, n readers` 这种一对多通信模式。

writer 和所有 reader 都会使用同一个共享内存地址来初始化自己的 `ShmRingBuffer`，而数据读取位置通过 `current_idx` 控制。

共享内存读写过程如下。

读取过程：

1. `reader_x` 发现 `current_idx` 为可读状态，即状态 2 或状态 3。
2. 将 `data[current_idx]` yield 给 reader。
3. 读取完毕后，将 `metadata[current_idx][reader_x]` 从 0 改为 1，即已读。此时 `metadata[current_idx]` 处于状态 3 或状态 4。
4. `current_idx = (current_idx + 1) % max_chunks`，继续自旋。

写入过程：

1. writer 发现 `metadata[current_idx]` 为可写状态，即状态 1 或状态 4。
2. 将 `metadata[current_idx][0]` 的状态改为 0，如此状态 1 或状态 4 全部转为状态 1。
3. 将 `data[current_idx]` yield 给 writer。
4. 写入完成后，将 `metadata[current_idx][1:]` 全部置为 0，再将 `metadata[current_idx][0]` 置为 1，即状态 2。
5. `current_idx = (current_idx + 1) % max_chunks`，继续自旋。

其中类似 `data[current_idx]` 的表述是为了方便理解，底层需要通过 `data_offset`、`metadata_offset`、`current_idx` 和步长来确定。

写入过程中第 4 点的执行顺序至关重要。如果反过来，先将 `metadata[current_idx][0]` 置为 1，`metadata[current_idx]` 将立刻处于状态 3。假如置 0 操作还没有到第 `n - 1` 个位置，而 `n - 1` 对应的 reader 访问过来发现 `metadata[current_idx][n - 1]` 为 0，就会先读取一遍数据。随后，置 0 完成后，`n - 1` 对应的 reader 因为自旋再次访问，又读取一次数据，造成数据重复读取与混乱。

上述读写过程只是说明逻辑上 `ShmRingBuffer` 如何自旋。`ShmRingBuffer` 本身只是提供自旋能力，它自己的方法只提供获取 `shared_memory` 对应区间的能力；真正的自旋控制需要到 MessageQueue 运作机制中才能看到。

#### 3.2.3 SpinCondition

无锁同步解决了线程间同步问题，但自旋机制仍然存在天然缺陷：当读写条件尚未满足时，reader 和 writer 会不断轮询 `ShmRingBuffer` 中的 metadata。这种轮询不会主动让出 CPU，即使队列暂时没有任何可处理的数据，reader 和 writer 对应线程仍会持续消耗 CPU 计算资源。

为了避免无意义的盲等待，需要一种机制在无需继续自旋时挂起对应线程，并在满足条件后及时唤醒。vLLM 通过 `SpinCondition` 对 reader 和 writer 的自旋过程进行协调，在保证推理延迟的同时降低 CPU 无效占用。

vLLM 应该是用 C++ 实现了一个 spinloop，通过 `SPINLOOP_EXT_ENABLED` 开启。本文由于技术有限，只探讨 Python 侧的 `SpinCondition` 实现。

`SpinCondition` 的关键组件如下：

- `local_notify_socket`：配合 notify 机制发送和接收新数据写入信号，唤醒正在休眠的进程读取新数据。
- `write_cancel_socket` 与 `read_cancel_socket`：用来强制唤醒 reader。如果 writer 永远不会再写数据，例如 EngineCore 正在退出、某个 Worker 崩溃、MessageQueue 要被关闭、Executor 要停止，而此时 reader 很可能处于休眠状态，那么 reader 会永远阻塞。因此需要 cancel 机制，在没有数据到来的情况下也能打断 reader 休眠。reader 一般不会长期阻塞，所以不需要 cancel 机制。
- `poller`：`zmq.Poller()` 类。该类使用操作系统底层的 I/O 多路复用机制，Linux 下通常是 `epoll_wait`，macOS 下是 `kqueue`，Windows 下是 `select`。执行 `poll()` 后会阻塞在当前线程，直到 poller 的 socket 接收消息。`SpinCondition` 中 `local_notify_socket` 和 `read_cancel_socket` 组成了 `self.poller`，在 wait 过程中承担唤醒线程的作用。
- `last_read` 与 `busy_loop_s`：`last_read` 通过 `time.monotonic()` 获得初始化时的时间，随后通过 `record_read` 方法更新。这两个属性用于 `wait` 执行时和当前时间进行比较，判断是否休眠。

这里有一个细节值得注意：除了 PUB 和 SUB 之外，writer 和 reader 的 `local_notify_socket` 消息模式还有其他不同：

- reader 的 `CONFLATE`：开启后，接收消息容量会被限制。vLLM 将容量限制为 1，防止 writer 多次发送消息。无论有多少消息，只要唤醒 reader 读取即可，所以只需要一次，节省资源。
- writer 的 `SNDHWM`：是 Send High Water Mark 的缩写，即发送高水位线，对应发送队列最多能积压多少消息。原因同样是只要有一条消息就能唤醒 reader，不需要发送太多。

`SpinCondition` 协调的核心是 `wait`，典型流程如下：

1. 通过 `current_time = time.monotonic()` 获取当前时间，然后比较 `current_time` 和 `self.last_read + self.busy_loop_s`。
2. 如果 `current_time` 较小，vLLM 认为此时数据流量大，所以立即通过 `sched_yield` 释放 CPU 资源给其他线程执行推理过程。
3. 如果 `current_time` 很大，通过 `events = dict(self.poller.poll(timeout=timeout_ms))` 使当前线程休眠，直到以下事件到来。

可能唤醒线程的事件包括：

- `cancel`：上层调用 `SpinCondition.cancel()`，通过 `writer_cancel_socket` 发送消息给 `read_cancel_socket`，随后 `self.poller` 监听到该事件并唤醒线程。
- `notify`：上层调用 `SpinCondition.notify()`，通过 `local_notify_socket` 的 PUB/SUB 机制唤醒线程。这里要通过 `recv` 方法消费一次消息，否则后续消息将会无效，失去通知作用。
- 超时：同样唤醒线程。

注意这里只是唤醒线程，`SpinCondition` 不会做其他操作。事实上，`SpinCondition` 只是提供线程睡眠功能。至于唤醒线程后要做什么，是上层调用方的事情，和 `SpinCondition` 无关。因此需要等到 MessageQueue 的整体流程中，`SpinCondition` 的作用才会完整展示出来。

#### 3.2.4 MessageQueue 的完整广播机制

Socket 使用的是 ZMQ，这里网上有更详细的介绍，不再赘述。现在深入讨论如何将 `ShmRingBuffer`、`SpinCondition` 和 Socket 有机融合，实现前文提出的通信目标。

跨机通信通过 `remote_reader` 和基于 ZMQ 的 `remote_socket` 解决：

- writer 与 local reader：
  - 相对小的数据使用共享内存机制。共享内存使用同一个对象，`SpinCondition` 使用同一个通信地址，但角色略有不同，即 `is_reader` 是 `True` 还是 `False`。
  - 相对大的数据和控制流信号继续使用 ZMQ 作为底层通信组件。对于极大的数据，共享内存机制并不适用；同时一些同步信号也需要使用 Socket 传输。
- writer 与 remote reader：对于跨机传输，共享内存机制失效，因此退化为 ZMQ 实现。此时 local reader 数量，即 `n_local_reader`，为 0。

这里 vLLM 的实现体现了系统性能优化中的原则：Make the common case fast, and the rare case correct（让常见情况快到极致，让罕见情况保持正确）。

大多数情况下写入 `ShmRingBuffer` 的都是控制信息，数据体积并不大；一旦出现极端情况，就继续使用 Socket 传输数据。

不同角色的 MessageQueue 需要共享很多对象或地址，所以 writer 初始化后会通过 `export_handle` 将自身信息提取为 handle，reader 调用 `create_from_handle` 利用 writer 的 handle 创建自身。

根据最开始的要求，MessageQueue 之间的通信使用广播方式，即 `broadcast_object()`。

`enqueue` 和 `dequeue` 的流程较复杂，需要先了解几个重要步骤。

##### 3.2.4.1 序列化与反序列化

首先是序列化与反序列化。

vLLM 利用 Python PEP 574 引入的 Pickle 协议 5，支持带外数据（Out-of-band data），允许大型数据对象独立于主要 Pickle 数据流进行传输。

序列化时，vLLM 中设置小于 `1024 * 1024 = 1 MiB` 的数据直接序列化；大于等于这个值的数据则留在原来的内存中，Pickle 会用 `PickleBuffer` 封装这部分数据，然后将 `PickleBuffer.raw()` 返回的对应 `memoryview` 追加到 `all_buffers` 中。最后，将 Pickle 主字节流添加到 `all_buffers[0]` 位置上。

可以简单将 `memoryview` 对象理解为 PyTorch Tensor 的视图，即 `memoryview` 不存储任何数据，只是内存视图。

序列化完成后，按照既定协议，将 `all_buffers` 中的主字节流和 `memoryview` 转移到 `SharedMemory` 中以完成进程间通信。这里 `SharedMemory` 为 `ShmRingBuffer.buf`，这也是一个 `memoryview` 对象。`memoryview` 对象赋值时会将真实数据拷贝到目标 `memoryview` 上，也就是说，此时底层执行了 `memcpy`，将数据从 writer 进程拷贝到了共享内存中。

随后，reader 从同一个对象的 `SharedMemory` 中按照既定长度拿到数据的 `memoryview` 后，使用 `pickle.loads` 将 `memoryview` 对应数据复制到自己的内存空间。

注意到共享内存的第一个字节存储的是一个叫 `overflow` 的数据，它其实是一个标志位。这就引出了前文所述的数据实在过大的问题。

在 3.2.2 节中可以看到，`ShmRingBuffer` 创建的 `SharedMemory` 中 data 区域固定为 `max_chunk_bytes * max_chunks` 大小，意味着每次只能向 `SharedMemory` 写入 `max_chunk_bytes` 的数据，每写 `max_chunks` 次为一轮。如果序列化后发现要存储的数据长度已经超出 `max_chunk_bytes`，就会退化为使用 `local_socket` 传输。

reader 需要知道是用共享内存接收，还是用 `local_socket` 接收，所以 vLLM 在共享内存的第一个 Byte 位置固定了一个标志位。`overflow` 为 0 代表用共享内存，为 1 代表用 Socket。

当然，向 `ShmRingBuffer` 写入或读取数据时需要判断 metadata 状态。`enqueue` 中通过 `acquire_write` 获取写共享内存的资格，`dequeue` 中通过 `acquire_read` 获取读共享内存的资格。

##### 3.2.4.2 acquire_write 与 acquire_read

对于 `acquire_write`，如 3.2.2 节所述，需要 metadata 中的数据处于状态 1 或状态 4。在 `acquire_write` 中，vLLM 会检查当前状态。

这里有一个前文没有提过的 `memory_fence()`。简单来说，它通过获取和释放锁，利用底层计算原语控制执行顺序并保证数据可见性，具体实现可查阅源码。

`acquire_write` 的核心流程如下：

1. 使用 `while True` 保证不断重试。
2. 通过 `current_idx` 将对应的 metadata 区域提取为 `metadata_buffer`。
3. `check()` 检查 `metadata[current_idx]` 当前状态：
   - `written_flag` 为 0，对应状态 1。
   - `written_flag` 为 1 且读取标志位也全都是 1，对应状态 4。
4. 如果 `check()` 结果为 `False`，直接 `sched_yield()`，将 CPU 时间交给其他线程，直到下一次 CPU 时间。
5. 如果 `check()` 为 `True`，按照 3.2.2 节中的写入过程进行操作。

`acquire_read` 大体和 `acquire_write` 相同，不过有以下差异：

- 需要状态不同，所以 `check()` 实现不同。
- local reader 只会检查自己对应位置的 `read_flag`。总体状态要求是状态 2 或状态 3，即 `written_flag` 为 1 且 `read_flag` 为 0。
- 发现无需读取时，reader 会调用 `SpinCondition.wait` 判断是否需要睡眠。
- 读取完成后，更新状态不同：只需要更新自己位置的 `read_flag`，同时记录本次读取时间，用来判断下一次是否休眠。

##### 3.2.4.3 enqueue 和 dequeue 的完成过程

不在同一台物理机上的 Worker 使用 `remote_socket` 即可，这里不再赘述。

基于前文所述，现在可以梳理 `enqueue` 和 `dequeue` 的完整过程。当上层调用 `MessageQueue.broadcast` 后：

writer 执行 `enqueue` 方法：

1. 序列化数据。
2. 根据数据长度判断传输方式。
3. 如果 `total_bytes >= ShmRingBuffer.max_chunk_bytes`，则使用 `local_sockets.send_multipart`。
4. 如果 `total_bytes < ShmRingBuffer.max_chunk_bytes`：
   - 通过 `acquire_write` 获取写入资格。
   - 完成数据从 writer 进程到共享内存的拷贝。
   - 更新共享内存状态。
   - 通过 `SpinCondition.notify` 通知睡眠的 reader 读取数据。
5. 通过 `remote_sockets.send_multipart` 向远程 reader 发送。

reader 执行 `dequeue` 方法：

1. 通过 `acquire_read` 获取读取资格。如果当前无法读取，调用 `SpinCondition.wait` 判断是否休眠。
2. 接收数据：
   - `overflow` 为 1：通过 `local_sockets.recv_multipart` 接收。
   - `overflow` 为 0：反序列化，调用 `pickle.loads` 将共享内存中数据拷贝到 reader 进程。
3. 通过 `remote_sockets.recv_multipart` 接收远程数据。

### 3.3 计算节点内部的组网

此处开始进入 Worker 的初始化过程，其中最重要的两个过程为 `init_device()` 和 `_init_message_queues()`。

本节主要关注 `init_device()`，因为 `_init_message_queues()` 同时涉及 MessageQueue 和 GroupCoordinator，本文会在 3.5 节讨论。

同样，为了避免变成代码走读，本文仍会将重要逻辑提取出来进行讨论。

#### 3.3.1 重计算 local_rank

前面提到的 Rank 都是逻辑上的。在这个阶段，需要根据逻辑 Rank 将进程绑定到对应的物理 GPU 上。

根据前文，`local_world_size` 指的是 DP 中一个模型副本所需的 GPU 总数。如果单台物理机能够装得下两个模型副本，并且并行策略也确实这样设计，那么两个 Executor 对应的 Worker 的 `local_rank` 都会变成 `[0, 1, 2, 3]`，这样就会出现冲突。

为了避免这样的冲突，vLLM 在 `local_rank` 的基础上，将单台物理机内部的 DP Rank 配合 `tp_pp_world_size` 叠加上去。两组 Worker 的 `local_rank` 会变为：

```text
[0, 1, 2, 3, 0, 1, 2, 3]
  + [0, 0, 0, 0, 1, 1, 1, 1] * [4, 4, 4, 4, 4, 4, 4, 4]
= [0, 1, 2, 3, 4, 5, 6, 7]
```

这样刚好对应单台物理机的 8 个 GPU。

#### 3.3.2 绑定物理 GPU

首先看全局 GPU Rank 是怎么确定的。

在 `arg_utils.py` 中，`FlexibleArgumentParser` 会对用户命令行输入参数进行解析，其中 `device_ids` 代表希望参与此次推理的 GPU 范围。在 `ParallelConfig` 的构建过程中，它会通过 `_resolve_device_ids()` 赋值给 `assigned_physical_gpu_ids` 属性。

不过在不同情况下，`device_ids` 经过 `_resolve_device_ids()` 解析后的意义有所不同：

- `None`：用户没有提供该参数，`assigned_physical_gpu_ids` 直接为 `None`。
- 全是 `str`：
  - 可以 `int()`：返回对应的 `list[int]`。
  - 不可以 `int()`：说明是 UUID，通过 NVIDIA 提供的接口使用 UUID 找到 index，返回 `list[int]`。
- 全是 `int`：
  - 设置了 `CUDA_VISIBLE_DEVICES` 环境变量：这种情况下 `device_ids` 会被认为是索引，返回基于 `device_ids` 为索引、在 `CUDA_VISIBLE_DEVICES` 上取出的元素组成的 `list[int]`。
- 部分 `str`、部分 `int`：报错，因为不允许 UUID 和 int 混用。

最终，`assigned_physical_gpu_ids` 代表物理 GPU Rank。比如提供纯 int 的 `--device-ids` 为 `[1, 2]`，并且提供 `CUDA_VISIBLE_DEVICES` 为 `[2, 3, 4, 5]`，最终本次推理能够访问的 GPU 物理 Rank 为 `[3, 4]`。

先不急于获取逻辑 `local_rank` 对应的物理 GPU Rank，需要先明确一点：`torch.device(f"cuda:{index}")` 的 `index` 是什么。

这里的 `index` 并不是 GPU 的物理 Rank，而是当前进程可见 GPU 列表中的索引。例如当 `CUDA_VISIBLE_DEVICES` 为 `[2, 3, 4, 5]` 时，index 并不是 2 到 5，而仍然是 0 到 3。

在 `init_device()` 中，vLLM 调用 `current_platform.logical_device_id_to_visible_device_id(self.local_rank)` 完成逻辑到物理的映射。这个过程分两步。

第一步，获取 `local_rank` 对应的物理 Rank。根据 `assigned_physical_gpu_ids` 的不同值，会出现以下几种情况：

- `None`：用户没有设置 `--device-ids`。
  - `CUDA_VISIBLE_DEVICES` 不为 `None`：返回 `CUDA_VISIBLE_DEVICES[local_rank]` 作为物理 Rank。
  - `CUDA_VISIBLE_DEVICES` 为 `None`：直接返回 `local_rank` 作为物理 Rank。
- 非 `None`：返回 `_assigned_physical_gpu_ids[local_rank]` 作为物理 Rank。

第二步，在获取到物理 Rank 后，根据 `CUDA_VISIBLE_DEVICES` 决定最终可见设备索引：

- `CUDA_VISIBLE_DEVICES` 为 `None`：直接用物理 Rank 当作索引，最终结果为 `torch.device("cuda:rank")`。
- `CUDA_VISIBLE_DEVICES` 非 `None`：判断 Rank 作为索引的正确性，通过后得到 `torch.device("cuda:CUDA_VISIBLE_DEVICES[rank]")`。

例如 `local_rank` 为 1，根据前面的例子，该 Worker 对应的物理 Rank 为 `[3, 4]` 中索引为 1 的 GPU，最终结果为 `torch.device("cuda:4")`。

经过上述过程，可以得出一般结论：`CUDA_VISIBLE_DEVICES` 决定 Torch 能够访问的物理 GPU，`--device-ids` 决定在 Torch 可访问物理 GPU 基础上，根据索引还要进一步使用哪些 GPU。

#### 3.3.3 定制化 Kernel

定制化 Kernel 主要分为两个方面：批处理不变性和通信算子定制化。

首先理解批处理不变性要解决什么问题。批处理不变性也可以叫绝对确定性。在推理过程中，由于某些顺序的调换，可能导致结果不同。

数学上 `A + (B + C)` 与 `(A + B) + C` 满足加法结合律，但在 GPU 上做浮点数运算时，由于舍入差异会造成精度误差。一方面，推理过程中根据输入形状不同可能采取不同 Kernel 动态切换，不同算法内部循环和累加顺序可能不同；另一方面，为了让所有 SM 都不空闲，遇到大 batch 时，底层算法也经常会把矩阵进行 Split-K 处理。由于线程块调度不可预测，最后的累加和也会不可预测。

因此，vLLM 使用 `enable_batch_invariant_mode()` 和 `override_envs_for_invariance()` 两个方法处理批处理不变性：

- 用 Triton 重写核心算子，并用 `torch.library.Library().impl()` 方法对 PyTorch 底层函数进行拦截和替换。
- 针对不同显卡架构采取不同策略。例如针对 Ampere 架构替换 Kernel，而对于 Hopper 架构设置对应环境变量即可。
- 封死所有可能产生随机性的环境变量。

通信算子的定制化问题会在 3.4.3 节讨论。此处只需知道 vLLM 通过 `set_custom_all_reduce` 方法改变全局变量 `_ENABLE_CUSTOM_ALL_REDUCE` 为 `True` 或 `False`，来决定是否使用定制化 all-reduce 算子。

#### 3.3.4 通信进程组的建立

vLLM 调用 `init_distributed_environment()` 和 `ensure_model_parallel_initialized()` 两个方法完成所有通信进程组的建立。如果将两个过程合在一起看，整体流程如下：

1. 通过 `torch.distributed.init_process_group()` 建立全局通信基础。
2. 通过 `init_world_group()` 方法建立 `_WORLD` 通信子组。
3. 通过 `init_model_parallel_group()` 方法建立 `_INNER_DP_WORLD`、`_TP`、`_DCP`、`_PCP`、`_PP`、`_DP`、`_EP`、`_EPLB` 等并行策略相关通信子组。

在两个 init 方法中，本质上都是构建 `GroupCoordinator` 类，并在初始化过程中调用 `torch.new_group()` 创建通信子组。

`GroupCoordinator` 十分复杂且重要，因此会在第 3.4 节中单独讨论。既然 `GroupCoordinator` 用来构建不同并行策略的通信子组，就需要使用每个并行策略包含的 Rank。进入下一节前，可以先看 vLLM 如何巧妙地对不同并行组的 Rank 进行提取。

假如设置 `TP = 4, PP = 2, DP = 2`，逻辑上通信子组如下：

```text
_TP: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
_PP: [[0, 4], [1, 5], [2, 6], [3, 7], [8, 12], [9, 13], [10, 14], [11, 15]]
_DP: [[0, 8], [4, 12], [1, 9], [5, 13], [2, 10], [6, 14], [3, 11], [7, 15]]
```

其中 `_DP` 在推理阶段不需要通信，但训练时由于梯度更新需要全局数据，每个 DP 组内分到相同模型副本的 GPU 也要通信。

vLLM 使用 PyTorch Tensor 数据结构优雅地将这些 Rank 根据每个并行策略重组：

1. 构造一个形状和拓扑结构一致的 Tensor。
2. 根据每个并行策略对 Tensor 进行 `view()` 和 `unbind(0)` 操作。以 `_TP` 为例，可以抽取出每个 TP 组内连续、不同组作为元素的列表。
3. 将抽取出来的 `group_ranks` 转成 list，再构造成列表，用于通信组初始化。
4. 到哪个并行策略，就把哪个维度 `transpose()` 到最后一维，循环即可得到所有并行策略的 `group_ranks`。

### 3.4 GroupCoordinator

解决 Executor 与 Worker 间通信效率后，就要进入 Worker 到 Device 侧。

如果说 Executor 到 Worker 通信的难点在于各种限制条件，那么 Worker 到 Device 的难点则在于底层硬件复杂以及通信方式多样。

#### 3.4.1 主要功能

`GroupCoordinator` 承载的功能主要如下：

- 创建通信子组：根据 `group_ranks` 同时创建 CPU 后端（gloo）和 GPU 后端（如 NCCL）通信子组并进行管理。
- 定制化通信策略：对于不同场景和数据对象实现对应通信策略，以实现最大通信效率。

#### 3.4.2 通信子组的创建

这里需要构建 CPU 和 GPU 两种后端的子组，方便后续复杂数据拆分传输。PyTorch 提供了 `_create_subgroups_split_group` 方法，方便同时创建 CPU 和 GPU 子组。

#### 3.4.3 定制化通信策略

本文按照几种主要通信模式进行分类探讨：广播、集合通信和点对点通信。

##### 3.4.3.1 广播

一般来说，大模型推理过程中广播（broadcast）通信的特点如下：

- 数据内容：`input_ids`、`positions`、`attention_metadata` 等元数据。
- 数据结构：Tensor、Python Object、Tensor Dict 等。

vLLM 直接使用 `torch.distributed` 的 `broadcast` 传输 Tensor，使用 `broadcast_object_list` 传输 Python Object。但对于 Tensor Dict 这种结构复杂且体积大的数据内容，vLLM 为了实现最大传输效率，使用元数据和张量拆分传输的方式。

在 `broadcast_tensor_dict` 方法中，vLLM 调用 `_split_tensor_dict` 方法，将 Tensor Dict 按照 key 拆分成 Tensor 和 `TensorMetadata`（device、dtype、size），然后分别对这两个组成的列表进行传输。

Tensor 方面通过 `torch.distributed.broadcast` 方法，根据 Tensor 所在位置选择 CPU 侧后端或 GPU 侧后端进行传输。`TensorMetadata` 通过 `GroupCoordinator.broadcast_object` 方法，采用 CPU 侧后端进行传输。

同时，`GroupCoordinator.broadcast_object` 方法内部不仅依赖 `torch.distributed.broadcast_object_list`。如果是 TP 或 DCP（依赖 TP 实现，所以某些行为相同），还会优先选择 MessageQueue 类型进行传输，因为 TP 通信对延迟要求极高，所以尽可能使用共享内存。

运行时元数据的通信全貌如图所示：图中实线箭头代表调用方向，虚线箭头代表数据流向，既有实线又有虚线代表方向相同。蓝色为优先使用 MessageQueue 进行传输，红色为其他一般场景。

##### 3.4.3.2 集合通信

一般的集合通信主要为 `all_reduce`、`reduce_scatter` 和 `all_gather` 等方法。这类方法对性能要求极高，传统 `torch.distributed` 并不是大模型场景下的最优通信实现。因此，vLLM 使用 `device_communicator`（`DeviceCommunicatorBase` 及其子类），根据推理运行环境选择最快的通信后端。

拿 vLLM Attention 架构类比，`DeviceCommunicator` 类似于 `AttentionBackend`。

例如 `CudaCommunicator` 针对 `all_reduce` 会有相关决策。这里 CustomAllReduce 是否开启，就是利用 3.3.3 节提到的 `_ENABLE_CUSTOM_ALL_REDUCE` 环境变量。

vLLM 使用 PyTorch Dynamo 对推理进行优化。不过 Dynamo 的图编译过程生成的是纯计算图，意味着编译期只能处理基本 Tensor 或基本标量，例如 int、float、bool、string。如果碰到 Python 对象，它就无法正常解析，造成图断裂。

而在使用 `device_communicator` 的过程中，不免会经常使用到 `self`。这里 Dynamo 没有办法对 Python Object 进行序列化，即使可以，执行期这个对象也可能早就不存在。

为了让 Dynamo 完成图编译，vLLM 以 `all_reduce` 为例采用如下办法：

1. 将 `GroupCoordinator` 根据 `self.unique_name` 加入 `_groups` 字典中。
2. 注册一个接口算子到 `torch.ops.vllm`，这个算子只接收输入数据和一个代表 `GroupCoordinator.unique_name` 的字符串。
3. 在调用 `GroupCoordinator.all_reduce` 方法时，首先使用 `torch.ops.vllm.all_reduce`，也就是注册的接口算子。这样入参都是基本数据类型，Dynamo 会认为这是一个用户定制化算子，直接把它当作图中的黑箱节点，只关注输入和输出，不解析内部过程。
4. 在接口算子中，调用 `GroupCoordinator` 真正的 all-reduce 方法 `_all_reduce_out_place`，进而调用 `device_communicator.all_reduce`。

##### 3.4.3.3 点对点通信

对于 EP 的 dispatch 和 combine 通信，vLLM 直接使用 `device_communicator` 的相关通信算子。这部分由于本人知识限制，以及 MoE 适合单独成文，暂时略过。

这里直接来看点对点通信（P2P）。最经典的点对点通信是流水线并行不同 Stage 之间的通信，因此本文以此为例讨论 `GroupCoordinator` 的点对点通信。

在流水线并行下，每个 Stage 之间发送的数据可能是 Tensor Dict。和 `broadcast_tensor_dict` 类似，`isend_tensor_dict` 和 `irecv_tensor_dict` 也调用 `_split_tensor_dict` 方法将 Tensor Dict 拆分。拆分后的元数据使用 `send_object` 和 `recv_object` 通过 CPU 侧后端传输，Tensor 数据同样按照对应后端传输。

上一段提到的 `send` / `recv` 方法均带有字母 `i`，代表异步，均会返回 handle。同步的 `send_tensor_dict` 和 `recv_tensor_dict` 则是调用各自异步方法，获取 handle 后调用 `handle.wait()` 实现同步。

如果 `device_communicator` 也实现了对应的 `send_tensor_dict` 和 `recv_tensor_dict` 方法，在 `self.use_cpu_custom_send_recv` 为 `True` 的情况下，会使用 `device_communicator` 的方法。不过从标志位和代码来看，似乎只有纯 CPU 后端推理才会用 `device_communicator` 的 P2P 通信。

P2P 通信全貌如图所示：图中红色路径不使用 `device_communicator`，蓝色路径使用 `device_communicator`。

值得注意的是，在接收数据时，`send_tensor_dict` 比 `recv_tensor_dict` 多了一步 postprocess 操作，这是 all-gather 优化产生的。

一般模型较大的场景下，TP 组同台物理机部署，PP 组跨机部署。这与两种并行策略的通信量、NVLink 和 PCIe 传输速率差异有关。如果要传输的 Tensor 形状是 `world_size` 的倍数，可以将 Tensor 按照 `world_size` 进行拆分，并行组的每个 Rank 只传输某一个 shard，而在接收数据的单机通过 NVLink 和 all-gather 将 Tensor 还原回原始形状。这样可以极大减少 PCIe 阶段的数据量。

当然，并不是所有数据都值得用这种方式传输，vLLM 会通过 `_should_use_all_gather` 对 Tensor 进行选择。

### 3.5 Executor 与 Worker 的通信建立

在 3.1.3 节中，本文提到 Executor 与 Worker 通过 MessageQueue 进行通信；在 3.2 节中，本文阐述了 MessageQueue 的通信机制；在 3.4 节中，本文讨论了 `GroupCoordinator` 的职能。

了解这些前置概念后，就来到了最后一个步骤：Executor 与 Worker 的通信建立。代码执行顺序大致也是如此：先初始化 Worker，然后绑定 Device，最后建立 MessageQueue 通信机制。

首先，Executor 和 Worker 通信是双向的：

- Executor 向 Worker 发送 `SchedulerOutput`。
- Worker 向 Executor 传回 `ModelOutput`。

对于开启了数据并行但通信组 GPU 都在单台物理机的场景，该过程较简单。Executor 先创建好 MessageQueue，通过 `export_handle` 输出 handle 给 Worker。Worker 在前面的工作完成后，使用该 handle 调用 `create_from_handle` 创建对应 MessageQueue。Worker 再创建 `self.worker_response_mq`，让 Executor 能拿到对象即可。

不过当 DP 跨机后，创建 MessageQueue 的过程就没有那么简单。下面分别讨论。

#### 3.5.1 跨机 DP 场景下 Executor 到 Worker 的通信

这部分过程主要在 `MessageQueue.create_from_process_group` 方法上。

需要记住一个事实：Executor 只能和本机 Worker 通信，跨节点通信全部借助于 `group_rank` 为 `writer_rank`（一般是 0）的 Worker 和其他跨机 Worker 通信。

可能有人会问，MessageQueue 不是有 `remote_socket` 吗？问题在于，要创建 MessageQueue 需要先把 Executor 的 MessageQueue handle 传过去；但想把 handle 传过去又需要 MessageQueue，这是互相矛盾的。

整个初始化过程如下：

1. 通过 `GroupCoordinator.cpu_group` 获取 `group_rank`、`group_world_size`、`global_ranks` 信息。
2. 调用 `in_the_same_node_as` 方法（详情见附录 4.4）获取通信组各个 Rank 是否和 `writer_rank` 在同一台物理机上。
3. 根据 Worker 的 `group_rank` 采取不同策略。

`writer_rank`：

- 使用 `external_writer_handle` 创建 MessageQueue，或者根据通信组构建 `n_reader`、`n_local_reader` 等信息创建 MessageQueue。
- 由于 Executor 已经创建了 MessageQueue，所以此处直接使用 Executor 的 handle 创建 MessageQueue。
- 创建完后，通过广播将 handle 传递给其他 Rank。

其他 Rank：

- 通过广播接收 `writer_rank` 传递过来的 handle。
- 构建自己的 MessageQueue。
- 如果是阻塞模式，需要等到所有 Rank 的 MessageQueue 初始化完成。
- 每个 Rank 返回自己的 handle。

#### 3.5.2 跨机 DP 场景下 Worker 到 Executor 的通信

这部分过程主要在 `MessageQueue.create_from_process_group_single_reader` 方法上。

同样，Worker 到 Executor 的通信中，需要一个 `reader_rank` 的 Worker 作为 Driver 来收集其他 Worker 的 `ModelOutput`。

整个初始化过程如下：

1. 通过 `assigned_physical_gpu_ids` 或 `current_platform.device_count()` 获取 `local_size`。
2. 根据当前 Worker 的 `rank` 计算判断是否和 `reader_rank` 在同一台物理机上。
3. 创建 MessageQueue，`n_local_reader = 1 if same_node else 0`。
4. `export_handle()`，然后将所有 Rank 的 handle gather 到 `reader_rank` 上。
5. 如果是阻塞模式，需要等到所有 Rank 的 MessageQueue 初始化完成。
6. 每个 Rank 的 Worker 返回自己的 handle，以及所有 Rank handle 组成的列表。

至于为什么 Worker 到 Executor 这里判断是否在单台物理机的方法和 Executor 到 Worker 不一样，本文没有找到符合逻辑或非常笃定的答案。只能说 `create_from_process_group` 中的方法更加健壮，`create_from_process_group_single_reader` 中的方法更依赖通信组中的 Rank 连续分配。

## 4. 附录

### 4.1 Spawn or Fork

vLLM 通过 `_maybe_force_spawn()` 函数判断用 `fork` 还是 `spawn` 方式创建新进程。`fork` 的效率更高，所以 vLLM 会尽可能使用 `fork`，除非遇到下面几个场景：

| 触发场景 | 底层技术原因 |
|---|---|
| 环境变量已显式指定 `VLLM_WORKER_MULTIPROC_METHOD == "spawn"` | 尊重开发者意图：用户或部署脚本已经明确要求使用 `spawn`，直接无条件放行，跳过后续所有安全检查。 |
| 运行在 Ray 集群的 Actor 中，即 `is_in_ray_actor()` | 集群上下文传递需要：在 Ray 分布式环境中，必须将 `RAY_ADDRESS`（GCS 集群地址）等环境变量传递给子进程，让子进程知道如何连接到 Ray 集群。`spawn` 能够更安全、干净地在新进程中初始化这些分布式上下文。 |
| 启用了 NUMA 绑核优化，即 `"--numa-bind" in sys.argv` | 底层执行劫持限制：NUMA 绑定通常依赖类似 `numactl` 的系统级指令，这在底层使用了可执行文件劫持技术。这种对进程物理属性的强行修改要求进程必须是从零开始构建的全新实体（`spawn`），而不能是主进程的内存镜像（`fork`）。 |
| CUDA 或 XPU 已经被提前初始化，即 `cuda_is_initialized()` 或 `xpu_is_initialized()` | 防死锁与状态崩溃：如果主进程已经和 GPU 建立连接，生成 Context 或启动隐藏的硬件辅助线程，此时执行 `fork` 会导致子进程丢失这些隐藏线程，并共享同一个 Context 句柄。子进程一旦调用 GPU 就可能引发 segmentation fault 或死锁。必须用 `spawn` 保证子进程 GPU 环境绝对纯净。 |
| 运行在 WSL，即 `in_wsl()` | 驱动层兼容性缺陷：在 WSL 环境下，NVIDIA 的显卡管理与监控库（NVML）与 Linux 原生 `fork` 系统调用存在已知底层兼容性问题。如果强制 `fork` 会导致驱动层异常，只能通过 `spawn` 绕过。 |

### 4.2 NUMA

NUMA（Non-Uniform Memory Access，非统一内存访问）是现代多路（multi-socket）或多芯片模块（multi-die）服务器架构中管理 CPU 和内存的机制。

过去的 UMA 架构中，早期计算机采用 UMA（统一内存访问，也叫 SMP），所有 CPU 通过一条总线共享所有内存。随着 CPU 核心数增加，总线成为性能瓶颈。

现代 NUMA 架构为了解决瓶颈，将物理内存划分给特定 CPU 物理插槽（Socket），CPU 和它直连的内存组成一个 NUMA 节点（NUMA Node）。

- 本地访问（Local Access）：CPU 访问自己节点内的内存，速度最快，延迟极低。
- 远程访问（Remote Access）：CPU 访问其他节点上的内存，必须通过互联通道，如 Intel UPI 或 AMD xGMI，这会显著增加延迟并降低带宽。

在拥有多张 GPU 的服务器中，PCIe 插槽或互联总线，如 NVLink 所在的 Switch，通常也归属于不同 NUMA 节点。如果不加限制，操作系统调度器可能会让控制 GPU0 的 Python 进程运行在 CPU1（属于 NUMA 节点 1）上，而 GPU0 实际物理连接在 NUMA 节点 0 上。这会导致频繁跨节点通信，严重拖慢数据传输速率。因此需要对 GPU 进行 NUMA 绑核。

绑核就是利用工具，如 Linux 下的 `numactl`，强制操作系统调度器做到以下两点：

- CPU 绑定（CPU Affinity）：强制特定进程只能在属于特定 NUMA 节点的 CPU 逻辑核上运行。
- 内存绑定（Memory Affinity）：强制特定进程只能在其所在 NUMA 节点上分配物理内存。

### 4.3 管道与文件描述符

管道（Pipe）是操作系统内核提供的一种基于内存的、用于进程间通信的单向数据流通道。管道本质上是操作系统在内存里开辟出的一块缓冲区。

文件描述符（File Descriptor，FD）是操作系统中用于代表一个打开的文件、设备、网络 Socket 或管道等 I/O 资源的抽象索引。根据“万物皆文件”的哲学，操作系统会把管道也伪装成文件。

当进程需要打开一个真实文件、创建一个网络连接或创建一个管道时，操作系统内核会在后台把复杂的物理设备或内存空间准备好，然后给进程一个简单的数字。后续进程读写这个资源时，只需要向操作系统提交对该文件描述符的读写操作，操作系统就会查阅文件描述符表，找到真实目标并执行对应操作。

Linux 上写 bash 等脚本时最常见的文件描述符是 0、1、2：

- 0（标准输入，stdin）：通常对应键盘。
- 1（标准输出，stdout）：通常对应屏幕终端，用于打印正常信息。
- 2（标准错误，stderr）：通常对应屏幕终端，用于打印报错信息。

#### 4.3.1 子进程 close() 对主进程的影响

由于拿到的只是文件描述符，相当于开门的钥匙。子进程扔掉了钥匙，对主进程没有任何影响。

#### 4.3.2 关闭多余文件描述符的必要性

管道的一个机制是：只有当一个管道所有写入文件描述符都被销毁时，持有读取文件描述符的进程才会收到文件结束信号（EOF，End of File）。

其中一个例子是，在主进程异常销毁后，如果其他子进程持有 `death_writer`，子进程就不会收到 EOF 异常，进而成为僵尸进程。

### 4.4 如何获得在同一节点的 Rank

本节是对 `in_the_same_node_as` 方法的详尽介绍，整个过程如下。

首先，获取拓扑结构。额外注意一点：当前进程组如果是 `ProcessGroup` 类型，device 侧一定要是 NCCL 通信。

随后，创建一个 tensor 用来记录相关信息。1 代表对应索引的 Rank 在同一节点，否则为 0。这里还需要核对暗号。跨机访问对应内存可能会直接出现系统错误，所以需要忽略报错。

接下来要根据是否为 `source_rank` 分别处理。

`source_rank` 执行：

```python
# create a shared memory segment
shm = shared_memory.SharedMemory(create=True, size=128)
assert shm.buf is not None, "Buffer was not created"
shm.buf[: len(magic_message)] = magic_message
torch.distributed.broadcast_object_list(
    [shm.name],
    src=ranks[source_rank],
    group=pg,
)
```

它会把数据写进对应内存，并将内存名字广播给通信组内部其他 Rank。

非 `source_rank` 执行：

```python
recv = [None]
torch.distributed.broadcast_object_list(
    recv,
    src=ranks[source_rank],
    group=pg,
)
name = recv[0]
```

也就是先阻塞等待内存名称传来。

随后打开共享内存并检查暗号：

```python
with patch(
    "multiprocessing.resource_tracker.register",
    lambda *args, **kwargs: None,
):
    shm = shared_memory.SharedMemory(name=name)

assert shm.buf is not None, "Buffer was not opened"
if shm.buf[: len(magic_message)] == magic_message:
    is_in_the_same_node[rank] = 1
```

如果内存能够正确访问并且信息也正确，就代表确实在同一台物理机，即同一节点。

这里的 `patch` 是一种妥协。Python 会对这块不是它创建的内存也进行追踪，当进程销毁时会对这块内存进行清理，可能出现意想不到的问题。所以为了安全，将注册器的参数全部临时设置为 `None`，让它不在此进程中追踪这块内存信息。

执行 `finally`，退出共享内存占用：

```python
finally:
    if shm:
        shm.close()
```

等待同步，并清理资源：

```python
torch.distributed.barrier(group=pg)

with contextlib.suppress(OSError):
    if rank == source_rank and shm:
        shm.unlink()
```

最后获取结果：

```python
torch.distributed.all_reduce(is_in_the_same_node, group=pg)
aggregated_data = is_in_the_same_node
return [x == 1 for x in aggregated_data.tolist()]
```

通过 `all_reduce` 返回全部结果。
