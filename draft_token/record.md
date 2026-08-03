02_vLLM_parallelism
vLLM  并行实现
软件版本：本文内容基于 vLLM 0.25.0 release。
硬件环境：除非特别声明，本文讨论的硬件及性能表现均基于 NVIDIA GPU 系列。
文章定位：本文非源码走读指南。文章侧重于架构与机制探讨，必要时提供部分代码细节。如需代码层面的细节，请查阅官方源码。
目标读者：本文适合有一定大模型推理基础，并且了解各种并行概念以及特点的开发者阅读。
在大模型（LLM）推理优化领域，并行计算技术通过在不同维度上解耦和拆分推理过程，提升了系统的吞吐量，降低了延迟。目前业界主流的并行策略包括张量并行（TP，Tensor Parallelism）、流水线并行（PP，Pipeline Parallelism）、上下文并行（CP，Context Parallelism）、数据并行（DP，Data Parallelism）以及专家并行（EP，Expert Parallelism）等。
本文旨在从代码层面探讨这些并行手段在 vLLM 框架中的落地实现。为保持行文聚焦，暂时剥离 KV Transfer、EP、无状态模型、模型权重加载及底层通信算子的实现细节，将核心实现聚焦于 TP、PP、CP 等并行组的管理与构建。
本文内容将围绕以下几部分展开：
硬件抽象结构：理解 vLLM 对物理硬件的逻辑建模。
并行配置解析：追踪 vLLM 对用户并行意图的解析链路。
并行实现：深入剖析 vLLM 支撑各种并行策略的实现，将 Executor -> Worker -> Device 的过程分为两大部分进行探讨，同时我们会穿插重要组件的介绍。
1 硬件抽象结构
如图所示，vLLM 通过严谨的分层架构，将业务逻辑与底层复杂的硬件拓扑成功解耦。为便于后续对并行机制的探讨，我们先对各层抽象做简要对齐：
Client 层（接入网关）：专注外部请求的收发，与底层计算逻辑物理隔离。
Core 层（调度逻辑）：持有 Scheduler 和 KVCacheManager，负责实现 Continuous Batching 和 PagedAttention。本层专职于请求级和 Token 级的逻辑调度，属于“硬件无关层”。
Executor 层（执行引擎）：核心的隔离层。向上为 Core 提供统一 API，避免底层复杂的异构硬件信息污染核心调度逻辑；向下负责解析并管理跨卡、跨节点的计算调度方案。
Worker 层（计算实体）：对 Device（GPU）的直接建模。负责拉起计算资源、初始化分布式通信后端，并在指定硬件上执行实际的计算流。
ModelRunner 层（模型抽象）：在 Worker 内部屏蔽各类 LLM 的网络结构差异。
vLLM 并行相关的核心逻辑（如多进程拉起、通信组划分、设备映射等），主要收敛于在由 Executor 到 Worker 的调度链路中。因此，本文后续的探讨全盘聚焦于 Executor 层与 Worker 层，暂时略过 Client、Core 与 ModelRunner 层。
2 并行配置解析
vLLM 使用 arg_utils.py 中的 create_engine_config() 构造 VllmConfig 作为推理过程当中的全局配置，在 VllmConfig 中的 ParallelConfig 即为并行相关的配置。
并行相关配置的核心主要有以下几点：
基础切分维度：tensor_parallel_size、pipeline_parallel_size、prefill_context_parallel_size、data_parallel_size 等等。
全局派生维度：world_size 和 world_size_across_dp 等等。一些由方法 __post_init__() 根据其他配置计算得来，另一些通过 Python 的 @property 机制通过计算得来。
MoE 与专家并行：enable_expert_parallel、enable_eplb 和 all2all_backend 等等。
运行时后端与拓扑：distributed_executor_backend、numa_bind、numa_bind_cpus 等等。
ParallelConfig 中还使用了 _validate_parallel_config() 方法进行防御性校验。
3 并行实现
在理清了硬件的抽象结构与配置解析逻辑后，本文将深入探讨 vLLM 并行策略的落地实现。vLLM 通过之前提到的 distributed_executor_backend 参数提供了对多种分布式后端的支持（如 Ray 和原生多进程机制）。为了聚焦于底层核心逻辑、剥离外部框架带来的复杂性，本文将以原生多进程（MultiprocExecutor）为后端探讨落地实现过程。
从宏观上看，整个并行策略实现的最终目标为：严格依据配置中预设的硬件拓扑结构，精准构建分布式进程组（Process Group），并为各个维度初始化底层的通信机制。我们将整个并行组的构建与通信过程拆解为两个核心阶段：
控制面的调度与下发（从 Executor 到 Worker）：主进程如何拉起子进程并分配物理计算资源。
计算节点内部的组网（从 Worker 到 GroupCoordinator）：各个物理节点如何建立通信域并完成分布式协调。
3.1 控制面的调度与下发
从 Executor 到 Worker，代码逻辑和业务逻辑的差别很大（光 Executor 的初始化就要经历 Executor -> WorkerProc.make_worker_process -> Worker.worker_main -> WorkerProc.__init__ -> WorkerWrapperBase.init_worker -> ...，各种跳转 ）。为了避免变成代码走读，本文将控制面的调度与下发变为逻辑上的四个切面，旨在对读者后续自行查看源码也有帮助。
不光是 vLLM，其他的框架在落地并行的实现的时候可能也需要围绕这四个切面来展开：
拓扑与角色：解决怎么分工。
资源隔离与硬件亲和性：解决怎么榨干硬件性能。
进程间通信：解决主进程与子进程如何通信。
生命周期维护：解决兜底与排错。
3.1.1 拓扑与角色
vLLM 通过以下 ParallelConfig 的属性确定集群拓扑：
硬件资源：
nnodes：集群里总共有几台物理机。
node_rank：当前代码跑在哪一台物理机上。
基础切分维度：
tensor_parallel_size：合作计算同一层神经网络的 GPU 数量。
prefill_context_parallel_size：共同处理同一个超长请求的 GPU 数量。
pipeline_parallel_size：组成一条完整的前向计算的 GPU 或 GPU 组数量。
data_parallel_size：全局范围内，拥有完整模型推理能力的副本总数。
data_parallel_size_local：单台物理机上运行的完整模型副本数量，至少为 1。该参数涉及的情况较多，例如多机还是单机，内部负载均衡还是外部负载均衡等多种组合情况，详情可见源码arg_utils.py。
全局派生维度：
world_size：指运行一个完整的模型副本需要多少 GPU，world_size = tensor_parallel_size * prefill_context_parallel_size * pipeline_parallel_size。
data_parallel_node_size：全局 DP 包含多少个 Nodes，data_parallel_node_size = data_parallel_size // data_parallel_size_local。
nnodes_within_dp：单个 DP 组内包含多少个 Node，nnodes_within_dp = nnodes // data_parallel_node_size
local_world_size：指单台物理机有多少 GPU 参与了一个模型副本，local_world_size = world_size // nnodes_within_dp。
单台物理机或者 GPU 的 Rank 计算：
node_rank_within_dp：在一个 DP 组内，单台物理机的 Rank。
global_start_rank：注意 global 指的是不是真正的全局 nnodes 维度，而是 world_size 维度，也就是一个 Node 里面的范围，所以 global_rank = local_world_size * node_rank_within_dp 
local_rank：[0, local_world_size]。
global_rank：global_rank = global_start_rank + local_rank。
由上可知拓扑相关的变量十分复杂，尤其是和 DP 相关的绕来绕去，包括命名也总是有歧义，所以我们用一个图来进行举例说明：
如图所示，本例中硬件信息和基础并行维度信息如下：
nnodes：4 台物理机。
node_rank：各机器为 [0, 1, 2, 3]。
tensor_parallel_size：8，一台单机上 8 个 GPU 全部参与 TP 并行组。
prefill_context_parallel_size：1，假设为 1，减少画图复杂度。
pipeline_parallel_size：2，两台物理机一起组成一个 PP 并行组。
那么全局派生维度为：
data_parallel_size_local：例子中单台机器是没有办法装下一整个模型副本的，需要两台机器组合流水线并行组才能完整地装下一个副本，所以 data_parallel_size_local 为 1。其实可以不按照装了多少个模型副本进行理解，可以按照这台机器参与多少个 DP 并行的 Rank 计算理解，这样 1 这个数字就比较直观了。
data_parallel_node_size：全局有多少个 Node 参与并行，两个大虚线方框。
nnodes_within_dp：那么一个 Node 里面有多少台机器呢，2 台，因为 nnodes = 4，意味着 4 台单机，然后 data_parallel_node_size= 2，意味着全局就 2 个 Node，那自然每个 Node 有 4 // 2 = 2 台单机。
world_size：这个没什么好说的，按公式算就好了，不过此例中 world_size 跨了两台物理机。
local_world_size：所以每个单台物理机中参与一个模型副本的 GPU 总数就用一个模型副本所需的 GPU 总数与一个 Node 里面有几台物理机做除法就好了，16 // 2 = 8。
Rank 计算：
node_rank_within_dp：一个 Node 里面两台机器，为 [0, 1]。
global_start_rank：world_size 为 16，跨了两台机器的 16 个 GPU，所以两台单机的  global_start_rank 为 [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8]。
local_rank：local_world_size 为 8，因此每台单机的  local_rank 为 [0, 1, 2, 3, 4, 5, 6, 7]。
global_rank：global_start_rank + local_rank = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] + [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]。
拓扑信息完成构建后，我们还要注意一些关键 Rank 的 GPU 或者物理机的角色，它们需要承担更多的职责：
node_rank_within_dp = 0：该物理机为主节点，主机点中主进程会创建全局广播消息队列，用来进程间，跨机器进程间通信。
global_rank % tensor_parallel_size == 0：各 TP 组的首节点，要承担数据汇聚和调度的职责。
3.1.2 资源隔离与硬件亲和性
这部分在 set_multiprocessing_worker_envs() 方法中。
3.1.2.1 资源隔离
我们知道进程的创建方法用 fork 和 spawn 之分。fork 创建的子进程对主进程的内存空间等资源直接全盘复制，效率较高，但是问题在于 内存空间的共享会造成一些致命问题。例如  CUDA Context 一旦建立起来以后，主进程和子进程共用的话会发生 Segmentation fault 或者死锁现象。而 spawn 是重新创建一整套资源，虽然效率低，但是胜在不会出现资源问题。因此为了保证效率但是又要正确地进行资源隔离，我们需要对 fork 还是 spawn 进行选择。对于 fork 还是 spawn 的判断方法可以查看附录 4.1 章节。
同时，PyTorch 底层使用了 OpenMP 来加速 CPU 上的计算（比如数据预处理、张量拼接）。默认情况下，PyTorch 会为每一个进程分配与机器物理核心数相等的线程来进行最大化的加速。这就意味着如果我们不是使用纯 CPU 环境进行推理，机器是 64 核 CPU，8 张 GPU，我们就会有 8 个 进程，每个进程里面有 64 个线程，总计 512 个线程。系统在运行的时候会发生频繁地上下文交换，严重占用 CPU 资源。在非 CPU 后端的推理环境中，我们需要限制每个进程只分配一个 CPU 线程，进行对 GPU 侧的控制即可。
3.1.2.2 NUMA 绑核与网卡绑定
NUMA (Non-Uniform Memory Access，非统一内存访问) 相关信息见附录 4.2。具体的 GPU、CPU、还有 NUMA Node的关联计算按照拓扑结构计算 rank 即可，这里源码更加清晰。
但还此处我们需要关注 vLLM 如何将命令行参数优雅地放进子进程的创建过程当中的：
首先构建不同的 numactl命令行参数（EngineCore 主进程一对多，Worker 子进程一对一）。
检测系统是否安装了 numactl 命令，同时获取项目中的 numa_wrapper.sh（具体查看源码）脚本路径。
将命令行参数存入特定的环境变量。
将默认的 Python 解释器替换为包装的脚本，因此在创建进程的时候默认的Python 命令从
变成下面
此过程全程都是 yield，在完成所有的进程创建操作后，执行 finally里面代码将执行环境全部还原。
GPU 需要绑定最近的网卡来实现跨节点传输的最大效率，方式就是根据 GPU 的逻辑索引定位到 GPU 的真实物理地址，然后找到最近的网卡地址进行绑定，在对应的环境变量里面进行更新。这部分大家可以查看源码。
3.1.3 进程间通信
这个章节我们会讨论 Excutor 和 Worker 之间的通信。一般来说，我们可以将通信分为两个部分：
控制面：用来完成进程的创建、同步与监测。
数据面：用来在推理过程中各进程间的信息同步。
关于数据面，由于使用到了 MessageQueue，我们需要在章节 3.2 进行讨论。
vLLM 使用最普通的管道（Pipe）来完成 Executor 与  Worker 控制面上的通信，如下图所示：
Executor 和每一个  Worker 均通过两个管道进行通信，分别是 Worker 用来通知 Executor 初始化已经完成的 ready 管道，和 Executor 用来通知 Worker 执行资源销毁的 death 管道。
所有的管道的 duplex 都置为 False，即单向传输。因为 Executor 只需要收 ready 管道的信息，发 death 管道的信息，而 Worker 只需要发 ready 管道的信息，收 death 管道的信息。同时，由于管道和文件描述符的机制（见附录 4.3），各进程应该 close() 自己不需要的进程，避免出现僵尸进程问题。例如 Executor 进程需要执行如下代码：
而对于 fork，情况会更加特殊。vLLM 使用循环对各个 Rank 的 Worker 进程进行创建，那么后续创建的 Worker 会继承创建前面 Worker 时的文件描述符，因此需要用一个  inherited_fds 整形列表来不停地追加前面 Worker 的文件描述符，然后在当前 Worker 创建过程中将这些描述符 close() 掉。如下图所示：
3.1.4 生命周期维护
Worker 进程会专门调用 monitor_death_pipe()，启动一个线程专门对 death_pipe 进行监听，直到 Executor 发送信息到 death_pipe 中，执行 shutdown()。
vLLM 在创建 Worker 进程的时候，还通过 signal.signal() 对操作系统级别的硬中断（Signal）进行拦截，如下所示：
signal_handler 是一个方法，是拦截到信号时的操作。SIGTERM 由操作系统、其他进程或系统管理员通过 kill 命令发送，而 SIGINT 由用户在终端输入中断字符（通常是 Ctrl + C）触发。vLLM 通过这种方式将操作系统级别的硬中断（Signal）转化为 Python 层面的异常，从而将进程进行正确销毁，防止 GPU 显存泄露、通信死锁等各种暴力停机可能会导致的问题。
用 threading.Event() 的 is_set() 方法来保证用户按多次 Ctrl + C 命令时只会抛出一次异常，避免重复操作。
为了保证 Executor 和 Worker 通信无误，vLLM 还设计了严格的同步策略，必须在所有的 Worker 返回 ready 信息并且通信方式也都全部就位以后，才会展开后续的工作。如下所示：
最后，为了方便定位问题，vLLM 使用 setup_proc_title_and_log_prefix() 为每个进程起了名字，在日志中可以明显看到是哪个 Rank的进程出了问题。
3.2 MessageQueue
3.2.1 Executor 与 Worker 通信的难题
我们不希望控制指令成为大模型推理时延中的瓶颈，因此 Executor 和 Worker 进程间通信的效率非常重要。
Python 自带的进程间通信本质上是点对点的通信模型，天生不支持广播。而 Executor 和所有 Worker 必须保持信息同步。如果用原生的 IPC，那就只能写个循环，每个 Worker 进程发送一遍。因此，我们需要引入“发布-订阅”或者“一对多广播”的通信模型，也就是消息队列。
同时，为了最大化的提高效率，我们希望能够使用共享内存这种方式来传递消息，避免像 Socket 一样还要拷贝到缓冲区再发送这样繁琐的操作。
不过使用共享内存的消息队列面临棘手的状态同步问题。Executor 必须确保队列中的消息已经被所有的 Worker 接收完毕，才能进行下一轮的覆盖写入。而为了解决这种同步问题，我们又必须引入锁这种机制。但是锁会极大地影响整体前向过程的推理性能，所以必须使用无锁算法来控制同步。
另外，如果模型过大，导致 Executor 和 Worker 不在同一台物理机上，共享内存就失效了，我们必须退回到跨机的 RPC 调用。
综上所述，我们需要一个远程时使用 Socket 通信，本地时使用共享内存通信，无锁同步的消息队列。那么去哪找这么一个数据结构呢？vLLM 的答案是自己造一个，即 vLLM MessageQueue。
本文首先会介绍 MessageQueue 的重要组件，然后将所有组件串联形成一个完整的广播过程。
3.2.2 ShmRingBuffer
ShmRingBuffer 是 MessageQueue 中用来实现共享内存和无锁自旋的组件。源码的注释也提供了 ShmRingBuffer 的图形介绍，我们这里从注释中抽象出来重点，用下图展示 ShmRingBuffer 的原理：
首先图中上半部分的方块代表着 ShmRingBuffer.shared_memory，作为 ShmRingBuffer 存储数据的容器，类型为 Python 内置的 SharedMemory。从 metadata 那一侧我们可以得出 ShmRingBuffer 仅针对 1 writer, n readers 这种一对多的通信模式。writer 和 所有 reader 都会使用同一个共享内存地址来初始化自己的 ShmRingBuffer，而数据读取的位置则通过 current_idx 进行控制。 
集合示意图的下半部分，我们解释下 shared_memory 的读写过程，解释下 ShmRingBuffer 是如何通过无锁自旋的方式实现一对多数据同步的：
读取过程：
reader_x 发现 current_idx 为可读状态（状态 2 或者状态 3）。
将 data[current_idx] yield 给 writer。
读取完毕后，将 metadata[current_idx][reader_x] 将从 0 改为 1，即已读。此时 metadata[current_idx] 处于状态 3 或者状态 4。
current_idx = (current_idx + 1) % max_chunks，自旋。
写入过程：
writer 发现 metadata[current_idx] 为可写状态（状态  1 或者状态 4）。
将 metadata[current_idx][0] 的状态改为 0，如此状态 1 或者 4 全部转为状态 1。
将  data[current_idx] yield 给 writer。
写入完成后，将 metadata[current_idx][1:] 全部置为 0，再将 metadata[current_idx][0] 置为 1，即状态 2。
current_idx = (current_idx + 1) % max_chunks，自旋。
其中类似于 data[current_idx] 的表述是为了方便，底层是需要通过 data_offset 和 metadata_offset，current_idx 还有步长来确定的。
写入过程中第 4 点的执行顺序至关重要。如果反过来，将 metadata[current_idx][0] 置为 1， metadata[current_idx] 将立刻处于状态 3。假如置 0 操作还没有到第 n - 1 个位置，而 n - 1 对应的 reader 访问过来发现 metadata[current_idx][n - 1]为 0 的话，那么会先读取一遍数据。随后，我们置 0 完成后，n - 1  对应的 reader 由于自旋又来访问了，再次读取数据，造成了数据的重复读取与混乱。如下图所示：
上述读写过程只是表明了逻辑上 ShmRingBuffer 是如何自旋的，但是 ShmRingBuffer 也只是提供了自旋的能力而已，它自己的方法中只提供了获取 shared_memory 对应区间的能力，真正的自旋控制需要到 MessageQueue 的运作机制中才能看到。
3.2.3 SpinCondition
无锁同步解决了线程间的同步问题，但自旋机制仍然存在一个天然的缺陷：当读写条件尚未满足时，reader 和 writer 会不断轮询 ShmRingBuffer 中的 metadata。而这种轮询不会主动让出 CPU，即使队列暂时没有任何可处理的数据，reader 和 writer 的对应线程仍会持续消耗 CPU 计算资源。为了避免这种无意义的盲等待现象，需要一种机制在无需继续自旋时挂起对应线程，并在满足条件后及时唤醒。vLLM 通过 SpinConition 对 reader 和各个writer 的自旋过程进行协调，在保证推理延迟的同时有效降低了 CPU 的无效占用。
vLLM 应该是用 c++ 实现了一个 spinloop，通过 SPINLOOP_EXT_ENABLED 开启。本文由于技术有限只会探讨 Python 侧的 SpinCondition 实现。
如下为 SpinCondition 的示意图：
local_notify_socket：配合 notify 机制，发送和接收新数据写入的信号，唤醒正在休眠的进程读取新数据；
write_cancel_socket 与 read_cancel_socket：用来强制唤醒 reader。如果 writer 永远不会再写数据怎么办？例如 EngineCore 正在退出、某个 Worker 崩溃，MessageQueue 要被关闭，Executor 要停止等等，而此时 reader 很可能处于休眠状态，这种情况下系统退出但是没有 reader 会写数据，reader 就永远处于休眠状态了。所以为了避免这种现象，需要 cancel 机制，在没有数据到来的情况下也能打断 reader 的休眠。至于 reader 一般不会长期阻塞，无需休眠，所以不需要 cancel 机制。
poller：zmq.Poller() 类，该类使用操作系统底层的 I/O 多路复用机制（Linux 下通常是  epoll_wait，Mac 下是 kqueue，Windows 下是 select），在执行该类的 poll() 方法后会阻塞在当前，直到 poller 的 socket 接收消息。SpinCondition 中 local_notify_socket 和 read_cancel_socket 组成了 self.poller，在 wait 过程中承担唤醒线程的作用。
last_read 与 busy_loop_s ：last_read 通过 time.monotonic() 获得初始化时的时间，随后通过 record_read 方法对自身进行更新。这两个属相用于  wait 执行时和当前时间进行比较，判断休眠与否。
这里有一个细节值得注意，除了 PUB 和 SUB 之外，writer 和 reader 的 local_notify_socket 的消息模式还有其他不同：
reader 的 CONFLATE：字面上是”合并、折叠“的意思，开启这个模式，接收消息的容量会被限制，vLLM 将容量限制为 1，防止 writer 多次发送消息，反正不管多少消息，只要唤醒 reader 读取就好了，所以只需要一次，节省资源；
writer 的 SNDHWM：是 SeND High Water Mark 的缩写，即”发送高水位线“。对应的，这个是发送队列最多能积压多少消息的限制，同样是因为只要有一条消息就能唤醒 reader，不需要发送太多；
 SpinCondition 协调的核心为 wait，一个典型的 wait 流程如下：
通过 current_time = time.monotonic() 获取当前时间，然后比较 current_time 和 self.last_read + self.busy_loop_s：
如果 current_time 较小，vLLM 认为此时数据流量大，所以立即通过 sched_yield 释放 CPU 资源给其他线程执行推理过程。
如果 current_time 很大，通过 events = dict(self.poller.poll(timeout=timeout_ms)) 使当前线程休眠，直到以下三种事件的消息到来：
注意这里只是唤醒线程，SpinCondition 不会做任何的操作。事实上 SpinCondition 只是提供了线程睡眠的功能，至于唤醒线程要做什么，那是上层调用的事情，跟 SpinCondition 无关。所以要等到 MessageQueue 的整体流程的时候，SpinConditoin 的作用才会展示出来。
cancel：上层调用了 SpinCondition 的 cancel 方法，通过 writer_cancel_socket 发送消息给 read_cancel_socket，然后 self.poller 监听到了此事件，唤醒线程。
notify：上层调用了 SpinCondition 的 notify 方法，通过 local_notify_socket 的 PUB 和 SUB 机制唤醒线程。这里要通过 recv 方法消费一次消息，否则后续的消息将会无效，失去了通知的作用。
超时，同样唤醒线程。
3.2.4 MessageQueue 的完整广播机制
Socket 使用就是 ZMQ，这里网上有更详细的介绍，不再赘述。我们现在来深入讨论如何将 ShmRingBuffer、SpinCondition 和 Socket 有机融合在一起，实现我们之前的通信目标。
以下为 MessageQueue 的总体架构。
之前讨论的问题还剩下一个跨机通信，就是通过 remote_reader 和基于 ZMQ 的 remote_socket 解决的：
writer 与 local_reader：
相对小的数据：使用共享内存机制。共享内存当然都是直接用的同一个对象，而 SpinCondition 用的是同一个通信地址，但是角色略有不同（is_reader 是 True 还是 False）；
相对大的数据和控制流信号：对于极其大的数据，共享内存机制并不适用，所以仍然使用 ZMQ 作为底层通信组件，同时一些同步信号也需要使用 Socket 传输；
writer 与 remote_reader：对于跨机传输，共享内存机制失效，因此退化为 ZMQ 实现，此时 local_reader 数量，即 n_local_reader，为 0；
这里 vLLM 的实现体现了系统性能优化中的一句名言：Make the common case fast, and the rare case correct（让常见情况快到极致，让罕见情况保持正确。)。
毕竟大部分的情况下存入到 ShmRingBuffer 的数据都是控制信息，数据体积并不大，但是万一出现了极端情况，就老老实实地继续用 Socket 传输数据。
不同角色的 MessageQueue 需要共享很多对象或者地址，所以在 writer 初始化后，通过 export_handle 将自身信息提取为 handle，reader 调用 create_from_handle 利用 writer 的 handle 来创建自身。
根据我们最开始的要求，MessageQueue 之间的通信均用广播的方式，即方法 broadcast_object()。
 enqueue 和 dequeue 的流程十分复杂，需要先了解重要步骤。
3.2.4.1 序列化与反序列化
首先是序列化与反序列化，如下图所示：
利用 Python PEP 574 引入 Pickle 协议 5，支持带外数据（Out-of-band data），允许大型数据对象独立于主要的 Pickle 数据流进行传输。
在序列化时，vLLM 中设置小于 1024 * 1024 = 1 MiB 的数据直接序列化，而大于等于这个值的数据则是留在原来的内存当中，Pickle 会用 PickleBuffer 封装这部分数据，然后将 PickleBuffer.raw() 返回的对应的 memoryview 追加到 all_buffers 中。最后将 Pickle 主字节流添加到 all_buffers[0] 位置上。
可以简单将 memoryview 对象理解为 PyTorch Tensor 的视图，意味着 memoryview 不存储任何数据，只是内存视图而已。
序列化完成后，按照既定协议，将 all_buffers 中的主字节流和 memoryview 转移到 SharedMemory 中以完成进程间通信。这里 SharedMemory 为 ShmRingBuffer.buf，这也是一个 memoryview 对象，而 memoryview 对象赋值的时候会将真实数据拷贝到目标 memoryview 上。也就是说，此时底层执行了 memcpy 命令将数据从 writer 进程中拷贝到了共享内存中。
随后，reader 从同一个 Object 的 SharedMemory 中按照既定长度将数据的 memoryview 拿到后，使用 pickle.loads 将 memoryview 对应的数据复制到自己的内存空间当中。
 注意到共享内存的第一个字节存储的是一个叫 overflow 的数据，它其实是一个标志位。这就引出了前文所述的数据实在过大问题。在 3.2.2 章节中可以看到 ShmRingBuffer 创建的 SharedMemory  中 data 区域是固定为 max_chunk_bytes * max_chunks 大小的，意味着每次只能向 SharedMemory 写入 max_chunk_bytes 的数据，每写 max_chunks 次为一轮。那么如果在序列化后发现要存储的数据长度已经超出了 max_chunk_bytes的话，就会退化使用 local_socket 进行传输。而 reader 那边需要知道用共享内存的方式接收还是用 local_socket 进行传输，所以 vLLM 在共享内存的第一个 Byte 位置上固定了一个标志位。 overflag 为 0 代表用共享内存，为 1 代表用 Socket。
当然，向 ShmRingBuffer 写入还是读取数据需要判断 metadata 的状态的。enqueue 中要通过 acquire_write 来获取写共享内存的资格，而 dequeue 中要通过 acquire_read 来获取读共享内存的资格。
3.2.4.2 acquire_write 与 acquire_read
对于 acquire_write，如 3.2.2 章节所述，需要 metadata 中的数据处于状态 1 或者 状态 4。在 acquire_write 中，vLLM 用如下代码检查状态：
这里有一个前文没有提过的 memory_fence()，简单来说它通过获取和释放锁，利用底层计算原语控制执行顺序以及保证数据可见性，具体的实现可以源码和自行查阅。
首先使用 while True 来保证不断重试，同时用 current_idx 将对应的 metadata 区域提取为 metadata_buffer。check() 就是在检查 metadata[current_idx] 当前的状态：
written_flag 为 0，对应状态 1。
written_flag 为 1 但是读取标志为也全都是 1，对应状态 4。
如果 check() 结果为 False 的话，就直接 sched_yield()，将 CPU 时间交给其他线程，直到下一次 CPU 时间。
如果 check() 为 True 的话，按照 3.2.2 章节中的写入过程进行操作：
 acquire_read 大体和 acquire_write 相同，不过有以下差异：
首先，因为需要状态的不同，check() 实现是不一样的
这里 local_reader 只会检查自己对应位置的 read_flag。而总体状态上要求是状态 2 或者状态 3，即 written_flag 为 1 但是 read_flag 为 0。
在发现无需读取时，reader 会调用 SpinCondition 用来判断是否需要睡眠
读取完成，更新状态不同，只需要更新自己位置的 read_flag，同时记录本次读取时间，用来判断下一次是否休眠
3.2.4.3 enqueue 和 dequeue 的完成过程
不在同一台物理机上的 Worker 使用 remote_socket 就可以了，不再赘述。
基于前文所述，现在我们可以梳理一下 enqueue 和 dequeue 的完整过程，当上层调用 MessageQueue.broadcast 后：
对于 writer 执行 enqueue 方法：
序列化数据。
根据数据长度判断传输方式
total_bytes >= ShmRingBuffer.max_chunk_bytes：local_sockets.send_multipart。
total_bytes < ShmRingBuffer.max_chunk_bytes：
通过 acquire_read 获取读取资格。
完成数据从 writer 进程到共享内存的拷贝。
更新共享内存状态
通过 SpinCondition.notify 通知睡眠的 reader 读取数据。
remote_sockets.send_multipart。
对于 reader 执行 dequeue 方法：
通过 acquire_read 获取读取资格。如果当前无法读取：调用 SpinCondition.wait 判断是否休眠。
接收数据：
overflow 为 1：local_sockets.recv_multipart。
overflow 为 0：反序列化，调用 pickle.loads 将共享内存中数据拷贝到 reader 进程。
remote_sockets.recv_multipart。
3.3 计算节点内部的组网
此处我们开始进入 Worker 的初始化过程当中，而这里面最重要的两个过程为 init_device() 和 _init_message_queues()。本章节我们主要关注 init_device()，因为 _init_message_queues() 同时涉及到 MessageQueue 和 GroupCoordinator，我们会在 3.5 章节对其进行讨论。同样，为了避免变成代码走读，我们仍然会将其中重要的逻辑提取出来进行深入探讨。
3.3.1 重计算 local_rank
前面我们提到的 Rank都是逻辑上的，在这个阶段我们要开始根据逻辑 Rank 将进程绑定到对应的物理 GPU上。根据前文我们知道 local_world_size 指的是 DP 中一个模型副本所需的 GPU 总数。那么如果单台物理机能够装得下两个模型副本，并请我们的并行策略确实是这样设计的，那么两个 Executor 对应的 Worker 的 local_rank 都会变成 ]0, 1, 2, 3，这样就会出现冲突。所以为了避免这样的冲突，vLLM 在 local_rank 的基础将单台物理机内部的 DP  Rank 配合 tp_pp_world_size 叠加上去，两组 Worker 的 locak_rank 就会变为 [0, 1, 2, 3, 0, 1, 2, 3] * [0, 0, 0, 0, 1, 1, 1, 1] * [4, 4, 4, 4, 4, 4, 4, 4] = [0, 1, 2, 3, 4, 5, 6, 7]，刚好对应上了单台物理机的 8  个 GPU。
3.3.2 绑定物理 GPU
首先我们来看一下全局的 GPU 的 Rank 是怎么确定的。
在 arg_utils.py 里面，FlexibleArgumentParser 会对用户的命令行输入参数进行解析，其中 device_ids 代表了我们想要参与此次推理的 GPU 的范围。在 ParallelConfig 的构建过程中，它会通过 _resolve_device_ids() 赋值给 assigned_physical_gpu_ids 属性。不过在不同的情况下，device_ids 经过 _resolve_device_ids() 解析后的意义有所不同：
None：用户没有提供这个参数，那么 assigned_physical_gpu_ids 直接为 None。
全是 str：
可以 int()：返回对应的 List[int].
不可以 int()：说明是 UUID，通过 NVIDIA 提供的接口使用 UUID 找到 index，返回 List[int]。
全是 int：
设置了 CUDA_VISIBLE_DEVICES 环境变量：这种情况下 device_ids 会被认为是索引，返回的是基于  devices_ids 为索引，在 CUDA_VISIBLE_DEVICES 上的元素组成的  List[int]。
部分 str，部分 int：报错，因为不允许 UUID 和 int 混用。
最终，assigned_physical_gpu_ids 代表的是物理 GPU 的 Rank，比如提供了纯 int 的 --device-ids 为 [1, 2] 并且提供了 CUDA_VISIBLE_DEVICES 为 [2, 3, 4, 5]，最终本次推理能够访问的 GPU 的 物理 Rank 为 [3, 4]。
先不急于获取逻辑上的 local_rank 对应的物理 GPU 的 Rank，我们先要明白一点，torch.device(f"cuda:{index}") 的 index 是什么。这里的 index 指的并不是 GPU 的 Rank，而是全局所有 GPU 中每个 Rank 的 GPU 的索引。例如当 CUDA_VISIBLE_DEVICES 为 [2, 3, 4, 5] 时，index 并不是 2 到 5，而是仍然是 0 到 3。
在 init_device() 中vLLM 调用 current_platform.logical_device_id_to_visible_device_id(self.local_rank) 来完成逻辑到物理的映射时要分两步，首先我们要获取 local_rank 对应的物理 Rank。根据前面的 assigned_physical_gpu_ids 不同值会出现以下几种情况：
None：用户没有设置 --device-ids，又分为以下两种情况：
CUDA_VISIBLE_DEVICES 不为 None：返回 CUDA_VISIBLE_DEVICES[local_rank] 作为物理 Rank。
CUDA_VISIBLE_DEVICES 为 None：什么都没有设置，那就直接返回 local_rank 作为物理 Rank。
不为 None：返回 _assigned_physical_gpu_ids[local_rank] 作为物理 Rank。
其次，在获取到物理 Rank 以后，根据 CUDA_VISIBLE_DEVICES：
None：直接用物理 Rank 当做索引，最终结果为 torch.device("cuda: rank")。
不为 None：判断 Rank 作为索引的正确性，通过后结果为 torch.device("cuda: CUDA_VISIBLE_DEVICES[rank]")。
例如 local_rank 为 1，那么根据前面的例子，该 Worker 对应的物理 Rank 为 [3, 4] 中索引为 1 的 GPU，最终结果为 torch.device("cuda: 4")。
经过上述过程的详解，我们可以得出以下一般结论：CUDA_VISIBLE_DEVICES 决定了 Torch 能够访问的物理 GPU，--device-ids 决定了在 Torch 能够访问的物理 GPU 基础上根据索引能够访问哪些 GPU。
3.3.3 定制化 Kernel
定制化 Kernel 主要分为两个方面：批处理不变性和通信算子定制化。
首先要理解批处理不变性要解决什么问题。批处理不变性，也可以叫做绝对的确定性，从名字可以看出，在推理过程中由于某些顺序的调换，可能会导致结果的不同。数学上 A + (B + C)  与 (A + B) + C 是满足加法结合律的，但是在 GPU 上做浮点数运算时，由于舍入的差异会造成精度上的误差。而推理过程中，一方面根据输入的形状不同可能会采取不同的 Kernel 进行动态切换，算法不同，内部循环和累加的顺序可能会不同，另一方面，为了让所有的 SM 都不空闲，遇到大 Batch 时，底层算法也经常会把矩阵进行 Split-K 处理，那么由于线程块的调度是完全不可预测的，最后的累加和也会不可预测。
因此 vLLM 使用了 enable_batch_invariant_mode() 和 override_envs_for_invariance() 两个方法对批处理不变性进行了处理。
用 Triton 重写了核心算子，并且用 torch.library_Library().impl() 方法对 PyTorch 的底层函数进行了拦截和替换。
针对不同的显卡架构，例如 针对  Ampere 架构 是对 Kernel 进行替换，但是对于 Hopper 架构，设置对应的环境变量即可。
封死所有可能产生随机性的环境变量。
通信算子的定制化问题我们到 3.4.3 章节再进行讨论，此处我们只要知道 vLLM 通过 set_custom_all_reduce 方法改变全局变量 _ENABLE_CUSTOM_ALL_REDUCE 为 True 活 False 来决定是否使用定制化的 all_reduce 算子。
3.3.4 通信进程组的建立
vLLM 调用 init_distributed_environment() 和 ensure_model_parallel_initialized() 两个方法来完成整个所有通信进程组的建立。如果我们将两个过程合在一起看的话，整个流程是这样的：
通过 torch.distributed.init_process_group() 建立全局通信基础。
通过 init_world_group() 方法建立 _WORLD 通信子组。
通过 init_model_parallel_group() 方法建立 _INNER_DP_WORLD、_TP、_DCP、_PCP、_PP、_DP、_EP、_EPLB 等并行策略相关的通信子组。
在两个 init 的方法里，本质上都是构建了 GroupCoordinator 类，在初始化的过程中调用 torch.new_group() 方法创建了通信子组。该类十分复杂但是重要，因此我们在第 4 章中单独讨论。GroupCoordinator 既然是用来构建不同并行策略的通信子组的，那么肯定要使用每个并行策略包含的 Rank，而在进入下一章前，我们可以看一下 vLLM 是如何巧妙地对于不同并行组的 Rank 是进行提取的。
假如我们设置 TP = 4, PP = 2, DP = 2 的一个并行策略，那么逻辑上通信子组将是下面的状况：
_TP：[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]。
_PP：[[0, 4], [1, 5], [2, 6], [3, 7], [8, 12], [9, 13], [10, 14], [11, 15]]。
_DP：[[0, 8], [4, 12], [1, 9], [5, 13], [2, 10], [6, 14], [3, 11], [7, 15]]，这里 _DP 在推理阶段是不需要通信的，但是训练的时候由于梯度更新需要全局的数据，所以每个 DP 组内分到模型副本相同的 GPU 也是要通信的。
如何优雅地将这些 Rank 根据每个并行策略进行重组呢？vLLM 使用了 PyTorch 提供的 Tensor 数据结构：
首先构造一个形状和拓扑结构一样的 Tensor
根据每个并行策略对 Tensor 进行 view() 和 unbind(0) 操作，以 _TP 为例
然后将抽取出来的 group_ranks 变成 list 再构造成列表，我们就拿到了每个 TP 组内连续，不同组作为元素的列表
然后就可以用 group_ranks 进行初始化了
到哪个并行策略就把哪个维度给 transpose() 到最后一维，往复循环就能得到所有并行策略的 group_rank
3.4 GroupCoordinator
解决了 Executor 与 Worker 间通信的效率后，我们就要进入到 Worker 到 Device 侧，如果说 Executor 到 Worker 通信的难点在于各种各样的限制条件，那么 Worker 到 Device 的难点则是底层硬件的纷繁复杂以及通信方式的多种多样。
3.4.1 主要功能
GroupCoordinator 承载的功能主要如下：
创建通信子组根据 group_rank 同时创建 CPU 后端（gloo）和 GPU 后端（如 NCCL）通信子组并进行管理。
定制化通信策略对于不同的场景和数据对象实现了对应的通信策略以实现最大通信效率。
3.4.2 通信子组的创建
这里我们要构建 CPU 和 GPU 两种后端的子组，方便后续复杂数据的拆分传输，而PyTorch 提供了 _create_subgroups_split_group 方法，方便我们同时创建 CPU 和 GPU 子组。
3.4.3 定制化通信策略
我们将按照集中主要的通信模式进行分类探讨：广播、集合、点对点。
3.4.3.1 广播
一般来说，大模型推理过程中广播（broadcast）通信的特点如下：
数据内容：input_ids、positions、attention_metadata 等元数据
数据结构：Tensor、Python Object，Tensor Dict 等
vLLM 直接使用  torch.distributed 的 broadcast（传输 Tensor）、broadcast_object_list（传输 Python Object），但是对于 Tensor Dict 这种结构复杂且体积大的数据内容，vLLM为了实现最大传输效率，使用了元数据和张量拆分传输的方式。
在 broadcast_tensor_dict  方法中，vLLM 调用  _split_tensor_dict 方法，将 Tensor Dict 按照 key 拆分成 Tensor 和 TensorMetadata（device、dtype、size()），然后分别对这两个组成的列表进行传输。Tensor 方面通过 torch.distributed.broadcast 方法，根据 Tensor 所在位置选择 CPU 侧后端或者 GPU 侧后端进行传输，而 TensorMetadata 通过 GroupCoordinator.broadcast_object 方法，采用 CPU 侧后端进行传输。
同时，GroupCoordinator.broadcast_object 方法内部不仅仅依赖于 torch.distributed.broadcast_object_list，如果是 TP 或着 DCP （依赖于 TP 实现，所以某些行为相同），还会优先选择 MessageQueue 类型进行传输，因为 TP 通信对延迟要求极高，所以尽可能使用共享内存。
运行时元数据的通信全貌如图所示：
图中实线箭头代表调用方向，虚线箭头代表数据流向，既有实线又有数据代表方向相同。蓝色为优先使用 MessageQueue 进行传输，红色为其他一般场景。
3.4.3.2 集合通信
一般的集合通信主要为  all_reduce、reduce_scatter 和 all_gather 等方法。这类方法对性能的要求极高，传统的 torch.distributed 并不是在大模型场景下的最优通信实现，因此 vLLM 使用了 device_communicator（ DeviceCommunicatorBase 及其子类），根据推理运行的环境选择最快的通信后端。拿 vLLM Attention 架构类比，DeviceCommunicator 类似于 AttentionBackend 。
例如 CudaCommunicator 针对 all_reduce 就会有如下的决策：
这里 CustomAllReduce 是否开启就是利用 3.3.3 章节我们提到的 _ENABLE_CUSTOM_ALL_REDUCE 环境变量。
vLLM 使用 PyTorch Dynamo 对推理进行优化，不过 Dynamo 的图编译过程生成的是纯计算图，意味着它有个硬性规定：编译期它只能处理基本的 Tensor 或者基本标量（int，float，bool，string）。如果碰到 Python 对象，它就无法正常解析，造成图断裂。
而在使用 device_communicator 的过程中，不免会经常使用到 self，例如：
这里 Dynamo 没有办法对 Python Object 进行序列化，就算可以，执行期这个对象可能早都已经不存在了。
为了让 Dynamo 能够完成图编译，vLLM 采用了如下办法，以 all_reduce 为例：
将 GroupCoordinator 根据 self.unique_name 假如到 _groups 字典中。
注册一个接口算子到 torch.ops.vllm，这个算子只接收输入数据和一个代表了 GroupCoordinator.unique_name 的字符串。
在调用 GroupCoordinator 的 all_reduce 方法时，首先使用 torch.ops.vllm.all_reduce，也就是我们注册的接口算子。这样入参都为基本数据类型，而 Dynamo 会认为这是一个用户定制化算子，直接会把它当做图中的一个黑箱节点，只关注它的输入和输出，而不会去解析它里面的过程。
如此，在接口算子中，我们调用 GroupCoordinator 真正的 all_reduce 方法，_all_reduce_out_place 方法，进而调用 device_communicator 的 all_reduce。
整个过程如图所示：
3.4.3.3 点对点通信
对于 EP 的 dispatch 和 combine 通信，vLLM 直接使用 device_communicator 的相关通信算子。这部分由于本人的知识限制以及 MoE 值得单开一篇文章的特殊性暂时略过。我们直接来看点对点通信（P2P）。而最经典的点对点通信是流水线并行不同 Stage 之间的通信，因此我们以此为例对 GroupCoordinator 的点对点通信进行讨论。
在流水线并行下每个 Stage 之间，他们发送的数据可能长这样：
我们发现其实它也是个 Tensor Dict，那么如同 broadcast_tensor_dict 的方法。isend_tensor_dict 和 irecv_tensor_dict 也调用 _split_tensor_dict 方法将 Tensor Dict 进行拆分。拆分后的元数据使用 send_object 和  recv_object 通过 CPU侧后端传输，Tensor 数据同样按照对应的后端进行传输。
大家应该注意到上一段我们提到的 send/recv 方法均带有字母 i，代表异步，均会返回 handle。而同步的 send_tensor_dict 和 recv_tensor_dict 就是调用各自的异步方法，获取 handle 后调用 handle.wait() 实现同步而已。
如果 device_communicator 也实现了对应的 send_tensor_dict 和 recv_tensor_dict 的方法，在 self.use_cpu_custom_send_recv 为 True 的情况下，会使用 device_communicator 的方法，不过从标志位和代码上来看，好像只有纯 CPU 后端推理才会用 device_communicator 的 P2P 通信。
P2P 通信全貌如图所示：
图中红色路径不使用 device_communicator，蓝色路径使用 device_communicator。
值得注意的是在接收数据的时候，send_tensor_dict 比 recv_tensor_dict 多了一步 postprocess 的操作，这是 all_gather 优化产生的。一般模型比较大的场景下，TP 组同台物理机部署，PP 组跨机部署，这是两种并行策略通信量还有 NVLink 和 PCIe 的传输速率区别导致的。但是要传输的 Tensor 的形状是 world_size 的倍数时，我们可以将 Tensor 按照 world_size 进行拆分，并行组的每个 Rank 只传输某一个 shard，而在接收数据的单机通过 NVLink 和  all_gather 将 Tensor 还原回原始形状，这样可以极大地减少 PCIe 阶段的数据量。如下图所示：
当然，并不是所有的数据都值得进行这种方式传输，vLLM 会对 Tensor 进行选择，如 _should_use_all_gather 的代码所示：
3.5 Executor 与 Worker 的通信建立
在 3.1.3 章节中我们提到了 Executor 与 Worker 通过 MessageQueue 进行通信，在 3.2 章节中我们详尽地阐述了 MessageQueue 的通信机制，在 3.4 章节中我们讨论了 GroupCoordinator 的职能。在了解完这些前置概念以后，我们就来到了最后一个步骤，Executor 与 Worker 的通信建立。事实上，代码执行的顺序也大概如此，需要先初始化 Worker，然后绑定 Device，最后建立 MessageQueue 的通信机制。
首先，Executor 和 Worker 通信是双向的。一方面 Executor 要向 Worker 发送 SchedulerOutput，另一方面 Worker 要向 Executor 传回 ModelOutput。
对于开启了数据并行，但是通信组的 GPU 都在单台物理机而言，该过程较简单。Executor 开始将 MessageQueue 创建好，通过 export_handle输出  handle 给 Worker。Worker 在前面的工作完成后，使用该 handle 调用 create_from_hanle 来创建对应的 MessageQueue。而 Worker 建个 self.worker_response_mq 让 Executor 能拿到对象就完事了。
不过当 DP 跨机了以后，创建 MessageQueue 的过程就没有那么简单了。接下来我们分别来看。
3.5.1 跨机 DP 场景下 Executor 到 Worker 的通信
这部分的过程主要在 MessageQueue.create_from_process_group 方法上。
我们需要记住一个事实：Executor 只能和本机的 Worker 进行通信，跨节点的通信全部借助于 group_rank 为 writer_rank（一般是 0）的 Worker 和其他跨机 Worker 进行通信。可能有人要问，MessageQueue 不是有 remote_socket 吗？问题在于，想要 MessageQueue 得先把 Executor 的 MessageQueue 的 handle 传过去，但是想要传过去又需要 MessageQueue，这是互相矛盾的。
整个初始化的过程如下：
首先通过 GroupCoordinator 的 cpu_group  获取 group_rank、group_world_size、global_ranks 信息
调用 in_the_same_node_as 方法（详情见附录 4.4）获取通信组各个 Rank 是否和 writer_rank 在同一台物理机上。
根据 Worker 的 group_rank 采取不同的策略：
writer_rank
使用 external_writer_handle创建 MessageQueue，或者根据通信组构建 n_reader、n_local_reader 等信息创建 MessageQueue。由于 Executor 已经创建了 MessageQueue，所以此处直接使用的是 Executor 的 handle 来创建 MessageQueue。
创建完后通过广播的形式将 handle 传递给其他 Rank。
其他 Rank
通过广播的方式接收 writer_rank 传递过来的 handle。
构建自己的 MessageQueue。
如果是阻塞模式下，要等到所有 Rank 的 MessagQueue 初始化完成。
每个 Rank 返回自己的 handle。
3.5.2 跨机 DP 场景下 Worker 到 Executor 的通信
这部分的过程主要在 MessageQueue.create_from_process_group_single_reader 方法上。
同样，Worker 到 Executor 的通信中，我们需要一个 read_rank 的 Worker 作为 Driver 来收集其他 Worker 的 ModelOutput。
整个初始化的过程如下：
首先，通过 assigned_physical_gpu_ids 或者 current_platform.device_count() 来获取 local_size。
根据当前 Worker 的 rank  通过如下计算判断是否和 reader_rank 在同一台物理机上：
创建 MessageQueue，n_local_reader = 1 if same_node else 0。
export_handle()，然后将所有 Rank 的 handle gather 到 reader_rank 上
如果是阻塞模式，需要等到所有 Rank 的 MessageQueue 初始化完成。
每个 Rank 的 Worker 返回自己的 handle，以及所有 Rank handle 组成的列表。
至于为什么 Worker 到 Executor 这里判断是否在单台物理机的方法和 Executor 到 Worker 的不一样，本文没有找到一个符合逻辑或者非常笃定的答案，只能说 create_from_process_group 中的方法更加健壮，create_from_process_group_single_reader中的方法更依赖于通信组中的 Rank 都是连续分配的。
4 附录
4.1 Spawn or Fork
vLLM 通过 _maybe_force_spawn() 函数来判断用 fork 还是 spawn 方式创建新进程，毕竟 fork 的效率还是高的，所以尽可能还是用 fork，除非碰上了下面几个场景：
触发场景 (代码条件)
对应的底层技术原因
环境变量已显式指定 VLLM_WORKER_MULTIPROC_METHOD == "spawn"
尊重开发者意志：用户或部署脚本已经明确要求使用 spawn，直接无条件放行，跳过后续所有安全检查。
运行在 Ray 集群的 Actor 中 is_in_ray_actor()
集群上下文传递需要：在 Ray 分布式环境中，必须将 RAY_ADDRESS（GCS 集群地址）等环境变量传递给子进程，让子进程知道如何连接到 Ray 集群。spawn 能够更安全、干净地在新进程中初始化这些分布式上下文。
启用了 NUMA 绑核优化 "--numa-bind" in sys.argv
底层执行劫持限制：NUMA 绑定通常依赖于类似 numactl 的系统级指令，这在底层使用了“可执行文件劫持（executable hijacking）”技术。这种对进程物理属性的强行修改，要求进程必须是一个从零开始构建的全新实体（spawn），而不能是主进程的内存镜像（fork）。
CUDA 或 XPU 已经被提前初始化 cuda_is_initialized() 或 xpu_is_initialized()
防死锁与状态崩溃：如果主进程已经和 GPU 建立了连接（生成了 Context 或启动了隐藏的硬件辅助线程），此时执行 fork 会导致子进程丢失这些隐藏线程，并共享同一个 Context 句柄。子进程一旦调用 GPU 就会引发 Segmentation fault 或死锁。必须用 spawn 保证子进程 GPU 环境绝对纯净。
运行在 WSL (Windows 上的 Linux 子系统) in_wsl()
驱动层兼容性缺陷：在 WSL 环境下，NVIDIA 的显卡管理与监控库（NVML）与 Linux 原生的 fork 系统调用存在已知的底层兼容性问题。如果强制 fork 会导致驱动层面的异常，只能通过 spawn 绕过。
4.2 NUMA
NUMA (Non-Uniform Memory Access，非统一内存访问) 是现代多路（Multi-socket）或多芯片模块（Multi-die）服务器架构中管理 CPU 和内存的机制。
过去的 UMA 架构： 早期计算机采用 UMA（统一内存访问，也叫 SMP），所有 CPU 通过一条总线共享所有内存。随着 CPU 核心数增加，总线成为了性能瓶颈。
现在的 NUMA 架构： 为了解决瓶颈，物理内存被划分给特定的 CPU 物理插槽（Socket），CPU 和它直连的内存组成一个 NUMA 节点 (NUMA Node)。
本地访问 (Local Access)： CPU 访问自己节点内的内存，速度最快，延迟极低。
远程访问 (Remote Access)： CPU 访问其他节点上的内存，必须通过互联通道（如 Intel UPI 或 AMD xGMI），这会显著增加延迟并降低带宽。
而在拥有多张 GPU 的服务器中，PCIe 插槽或互联总线（如 NVLink 所在的 Switch）通常也归属于不同的 NUMA 节点。 如果不加限制，操作系统的调度器可能会让控制 GPU0 的 Python 进程运行在 CPU1（属于 NUMA 节点 1）上，而 GPU0 实际上物理连接在 NUMA 节点 0 上。这会导致频繁的跨节点通信，严重拖慢数据传输速率。因此我们需要对 GPU 进行 NUMA 绑核。
绑核的过程： 绑核就是利用工具（如 Linux 下的 numactl）强制操作系统的调度器做到以下两点：
CPU 绑定 (CPU Affinity)： 强制特定进程只能在属于特定 NUMA 节点的 CPU 逻辑核上运行。
内存绑定 (Memory Affinity)： 强制特定进程只能在其所在的 NUMA 节点上分配物理内存。
4.3 管道与文件描述符
管道（Pipe）是操作系统内核提供的一种基于内存的、用于进程间通信的单向数据流通道。管道本质上是操作系统在内存里强行开辟出的一块缓冲区。
文件描述符（File Descriptor，FD）是操作系统中，用于代表一个打开的文件、设备、网络 Socket 或管道等 I/O资源的抽象索引。没错，根据“万物皆文件”的哲学，操作系统把管道也伪装成了文件。
当进程需要打开一个真实的文件，创建一个网络连接，或者创建一个管道时，操作系统内核会在后台把复杂的物理设备或内存空间准备好，然后丢给进程一个简单的数字。后续进程想读写这个资源，只需要对操作系统提交向文件描述符等于该数字的位置进行读写操作，操作系统就会查阅文件描述符表，找到真实的目标并执行对应的操作。
我们在 Linux 上写 bash 等脚本时最常见的文件描述符就是0，1，2。
0 (标准输入, stdin)：通常对应你的键盘。
1 (标准输出, stdout)：通常对应你的屏幕终端（打印正常信息）。
2 (标准错误, stderr)：通常对应你的屏幕终端（专门打印报错信息）。
4.3.1 子进程 close() 对主进程的影响
由于拿到的只是文件描述符，相当于开门的钥匙。那子进程扔掉了钥匙对主进程没有任何影响。
4.3.2 关闭多余文件描述符的必要性
管道的一个机制是：只有当一个管道所有的写入文件描述符都被销毁时，持有读取文件描述符的进程才会收到文件结束信号（EOF，End of File）。
其中一个例子，就是在主进程出现异常销毁后，如果其他子进程持有 death_writer，那子进程就不会受到 EOF 异常，进而成为僵尸进程。
4.4 如何获得在同一节点的 Rank
本章节是对 in_the_same_node_as 方法的详尽介绍，整个过程如下：
获取拓扑结构
额外一点，当前进程组如果是 ProcessGroup 类型的话 device 侧一定要是 NCCL 通信。
搞个 tensor 用来记录哪些相关信息，1 代表对应索引的 Rank 在同一节点，否则为 0，当然需要核对的“暗号”
跨机访问对应的内存可能会直接出现系统错误，所以这里我们需要将报错给忽略掉。
接下来要分 source_rank 与否
source_rank
1
2
3
4
5
# create a shared memory segment
shm = shared_memory.SharedMemory(create=True, size=128)
assert shm.buf is not None, "Buffer was not created"
shm.buf[: len(magic_message)] = magic_message
torch.distributed.broadcast_object_list([shm.name], src=ranks[source_rank], group=pg)
把数据写进对应的内存，并将内存的名字广播给通信组内部其他 Rank。
非 source_rank
1
2
3
recv = [None]
torch.distributed.broadcast_object_list(recv, src=ranks[source_rank], group=pg)
name = recv[0]
还是先阻塞等待内存名称传来。
1
2
3
4
5
6
7
8
with patch(
    "multiprocessing.resource_tracker.register",
    lambda *args, **kwargs: None,
):
    shm = shared_memory.SharedMemory(name=name)
assert shm.buf is not None, "Buffer was not opened"
if shm.buf[: len(magic_message)] == magic_message:
    is_in_the_same_node[rank] = 1
如果内存能够正确访问并且信息也正确，那就代表着确实在同一台物理机，即同一节点。这里的 patch 是一种妥协，因为 Python 会对这块不是它创造的内存也进行追踪，当进程销毁时会对这块内存也进行清理操作，这时就会出现意想不到的问题，所以为了安全，将注册器的参数全部临时设置为 None，让其不在此进程中追踪这块内存的信息。
执行 finally，退出共享内存的占用
1
2
3
finally:
    if shm:
        shm.close()
等待同步，并清理资源
1
2
3
4
5
torch.distributed.barrier(group=pg)
# ...
with contextlib.suppress(OSError):
    if rank == source_rank and shm:
        shm.unlink()
获取结果
1
2
3
4
torch.distributed.all_reduce(is_in_the_same_node, group=pg)
aggregated_data = is_in_the_same_node
# ...
return [x == 1 for x in aggregated_data.tolist()]
通过 all_reduce 返回全部结果。

