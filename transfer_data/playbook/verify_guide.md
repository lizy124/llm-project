# pool_verify — KV 池化 E2E 验证完整指南（通用层，agent 必读）

> 本文档由 `pool_verify/` 目录下的 README + 0/1/1b/2/3/4/5/6 号文件整合而成，**单一入口、可直接照做**。
> 源文件全部保留，本文档是"池化验证通用规则层"的合并视图。
>
> - **定位**：[env_install/](../env_install/) 解决"环境装好"（镜像→容器→网络→vllm/vllm-ascend 安装），本指南解决"**池化验好**"——mooncake 如何拉起、vllm 池化场景如何启动、测试怎么设计、E2E 什么才算通过
> - **适用范围**：vllm-ascend 池化相关特性开发 / 重构的 PR 验证（KV pool、AscendStoreConnector、mooncake/memcache backend、layerwise 等），跨服务器复用（51 / 112 / …）
> - **当前覆盖范围（N5 声明，2026-09-01 更新）**：本库核心覆盖 **kv_both（PD-Mixed）单实例**路径；**PD 分诊（kv_producer/kv_consumer + MultiConnector + proxy_server）已有单机 8 卡实测**（PR15367 S1，165 服务器：P TP=4 + D TP=4 + layerwise proxy，5/5 请求 + 27 层 LayerMetadata 完整 + 无 AttributeError 判据，详见 §5.5b）——多机 PD / 大规模 PD 仍未覆盖，后续按官方 kv_pool §2.3/§3.5 + layerwise 专用 proxy（`/v1/metaserver`，`--host` 禁 0.0.0.0）扩展
> - 后端包安装（mooncake-transfer-engine-npu / memfabric-hybrid / memcache-hybrid）→ [../env_install/7_kv_backends_install.md](../env_install/7_kv_backends_install.md)

---

## 目录

| 序号 | 阶段 | 文件（源） | 一句话 |
|------|------|-----------|--------|
| 0 | 总览 | README.md | 定位、结构、经验来源、维护规则、一页速查 |
| 1 | 概念速查 | 0_concepts.md | KV Pool 底层概念（DMA/SDMA/RDMA/RoCE/HCCS/HBM/Fabric Mem/MemFabric/MemCache/大页）与 protocol 选型 |
| 2 | mooncake 拉起 | 1_mooncake_startup.md | master 端口/配置/日志/kill/探活规范 |
| 3 | memcache 拉起 | 1b_memcache_startup.md | MetaService 独立启动/两份 conf/环境变量/layerwise 参数面/指标观测差异/探针 |
| 4 | vllm 池化启动 | 2_vllm_launch.md | 关键参数/启动脚本模板/READY 等待 |
| 5 | 测试方法 | 3_test_method.md | 长 prompt 门槛/验证矩阵/并发流程/对照实验/PD 分诊与 layerwise 正计数判据 |
| 6 | E2E 判定 | 4_pass_criteria.md | 存/取/去重三维证据链 + 虚假通过防范（正计数+跨进程证人+阴性对照）+ 重构双轮夹逼 + 验收表模板 |
| 7 | 判定误区 | 5_pitfalls.md | 哪些"看起来失败"其实正常，反之亦然 |
| 8 | 已知硬限 | 6_known_limits.md | DSV4 NSA multi-spec、指标口径、不可构造路径 |

**典型验证流程**：环境装好（env_install 0→6）→ 读第 2/3 章拉起存储后端（mooncake 或 memcache）→ 按第 4 章启动 vllm → 按第 5 章设计测试 → 按第 6+7 章判定 → 对照第 8 章确认失败是否属已知硬限。

---

## 0. 总览

### 0.1 经验来源

本库从以下已完结验证提炼（一手记录在这些目录，细节可溯源）：

| PR | 主题 | 一手记录 |
|----|------|----------|
| #14465 | DSV4 KV Pool（mooncake/memcache × layerwise 4 组矩阵） | map_51/pr14465_dsv4_kvpool/record_final/ |
| #14912 | kv_metrics_observability（4 指标，sync/layerwise/async 三路径） | map_51/pr14912_kv_metrics/kv_metrics_observability/HANDOVER.md |
| #15307 | GVA 线程收敛（进行中） | map_51/pr15307_gva_threads/PLAN.md |
| #15367 | layerwise 协议返工（双轮夹逼复验 + 虚假通过防范方法论，含 PD 分诊 S1 单机 8 卡实测） | map_165/record_final/*_20260901_rebase/ + ../refactor/layerwise/archive/test/e2e-report-20260901-rebase.md |

### 0.2 维护规则（与 env_install 一致）

1. **新 lesson 必须回写**：任何服务器上池化验证的新经验（新坑、新判定标准、新硬限），完成后回写本库对应编号文件（**同时同步更新本合并指南**）。
2. **服务器特定信息不放这里**：某台服务器的端口占用/权重路径/容器名等，放 `map_XX/README.md`（环境档案）。
3. **本库只放通用方法论**：换个服务器、换个 PR 依然成立的规则才入库。
4. 修改本库时同步检查 [AGENTS.md](../AGENTS.md) §6 已知坑清单是否需要更新。

### 0.3 一页速查（验证中最常用的 6 条）

1. mooncake master 端口一律 **50088**（50051 被宿主机 OceanStor DTMA 占用），metrics 默认 **9008**（本库脚本 `--metrics_port` 覆盖值；官方 mooncake 默认 **9003**，换机按实际启动参数对齐，勿照抄）。
2. vllm 必须加 **`--no-enable-prefix-caching`**，否则本地命中抑制 KV pool 存取。
3. 验证 prompt 必须 **≥ 2 个完整 granularity 块**（DSV4 granularity=4096 → prompt ≥ 8192 token），否则 `skip_save`。
4. **`master_put_start_requests_total=0` ≠ 池化未生效**（mooncake 走 batch API）；看 `master_allocated_bytes` / `master_key_count`。
5. E2E 通过 = **存 + 取 + 去重** 三维证据链齐全（详见第 6 章）。
6. kill mooncake 用显式 PID `kill -9`，`pkill -f` 不可靠。

---

## 1. 概念速查（KV Pool 底层背景）

> 池化验证知识库的**共享背景层**，被 mooncake / memcache / vllm 启动各章引用，**protocol 选型前先读**。
> 依据：官方 kv_pool.md / layerwise_kv_pool.md（附行号）；MemFabric 官方仓（gitcode.com/Ascend/memfabric_hybrid）与 MemCache 官方仓（gitcode.com/Ascend/memcache）README；未标行号者为通用硬件/网络常识。

### 1.1 核心概念表

| 概念 | 是什么 | 在池化场景中的角色 |
|------|--------|--------------------|
| **DMA** | 直接内存访问：数据搬运由硬件引擎完成，CPU 只发起请求 | 设备间所有拷贝的基础机制 |
| **SDMA** | 位于 die 上的设备侧 DMA 搬运引擎（跨 die 搬运走 HCCS，非"die 内部搬运"；NVIDIA 等价物：**Copy Engine**，`cudaMemcpyAsync` 底层）；SDMA 无官方全称展开（CANN 文档只定义 AI Core 的 MTE2/MTE3） | `device_sdma` 用它在 HCCS 上做片间拷贝；NVIDIA 侧对应 **P2P**（设备间直连拷贝：有 NVLink 走 NVLink，无则走 PCIe） |
| **RDMA** | 远程直接内存访问：绕过双方 CPU/内核直接读写远端内存 | 跨机 KV 传输的底层机制 |
| **RoCE** | RDMA over Converged Ethernet（基于融合以太网的 RDMA）：**一个协议标准**，把 RDMA 语义（远程直读直写、绕过 CPU）绑定到以太网上跑 | `device_rdma` **协议所走的物理链路**（device RoCE 网卡），用于 A2 及 A3+LINK_TYPE=ROCE；需 `HCCL_INTRA_ROCE_ENABLE=1` |
| **InfiniBand** | 原生 RDMA 专用网络（更贵的独立网络，RoCE 是它的以太网替代） | 参考对比；本验证不用 |
| **HCCS** | 华为 Cache Coherent System：同机内 NPU die 间高速互连总线（类比 NVLink）；**A2/A3 硬件都有**（A2 需 HCCS 款，另有 PCIe 款）。A3 走 HCCS 是**官方推荐**而非脚本默认——官方 env 脚本 4 处 `LINK_TYPE` 赋值**全部为 `"ROCE"`**（kv_pool.md:191/359/591/721），HCCS 只出现在 `elif` 分支条件；"A3 用 HCCS"的依据是 kv_pool.md:552 "recommended for A3 when HCCS is available" | **KV Pool 的 `device_sdma` 官方仅支持 A3**（A2 虽有 HCCS 硬件，但 KV Pool 走 RoCE/device_rdma）；`ASCEND_ENABLE_USE_FABRIC_MEM=1` 时把多卡内存池化为统一 fabric 视图 |
| **HBM** | 高带宽内存 = NPU 显存（device memory） | KV Pool 的**设备侧介质**：mooncake fabric 模式的 segment、memcache 多级池的 HBM 层。注意 memcache 的 `dram.size` / `medium="dram"` 指主机 DDR 层，与 HBM 不是同一层（kv_pool.md:553：A3 开 HCCS 时 dram 层置 0GB，恰证 dram≠HBM） |
| **Fabric Mem** | A3 上把多卡内存池化成统一设备内存视图（池化介质 **DRAM+HBM**，不只 HBM——MemFabric 官方口径）；关键在 HCCS 是 Cache Coherent（缓存一致）总线——任何 die 都像访问本地内存一样**直接直访**其他 die 的内存并自动维护一致性（是"直访"不是"搬运拷贝"），软件上呈现统一内存视图。术语桥接：vllm-ascend 文档口径叫 HCCS，MemFabric/硬件口径叫 Device **UB 1.0**（灵衢，同一套 fabric 的不同层叫法）；配套要求 LingQu Computing Network >= 1.5（kv_pool.md:1146） | A3 分支环境变量 `ASCEND_ENABLE_USE_FABRIC_MEM=1`；此分支下 mooncake `local_buffer_size` 不生效（见 §2.2）——统一编址直接直访，无需本地缓冲中转 |
| **MemFabric** | 昇腾开源的"**远程内存访问**"软件层（gitcode.com/Ascend/memfabric_hybrid，2025/11 开源）：让上层**像操作本地内存一样直接读写远端机器/其他卡的内存**——把各种搬运方向（卡↔卡、卡↔主机、跨机）统一成 memcpy 式接口（官方称 xcopy with GVA，即全局统一编址），底层走 RoCE 等由它管理 | **MemCache 的底层搬运底座**（官方明确 "MemCache depends on MemFabric"，kv_pool.md:495；MemCache README 称"多级内存和异构网络传输的底座"）。注意 `BACKEND_MEMFABRIC` **不是 KV Pool 后端**——KV Pool 后端注册表只有 mooncake/memcache/yuanrong（backend/__init__.py 的 backend_map）；`BACKEND_MEMFABRIC` 属 **SFA PD connector**（SfaRemoteD2HConnector，kv_p2p/sfa_pd_rd2h，其 `transfer_backend` 目前仅支持 `"memfabric"`），mooncake 不在该命名空间、有自己的传输引擎——三者属不同层次，不可说"与 mooncake 平级"。需先装 `memfabric-hybrid` 再 `memcache-hybrid`，运行时 source **两个** set_env.sh（/usr/local/memcache_hybrid/ 与 /usr/local/memfabric_hybrid/，layerwise_kv_pool.md:52-53） |
| **MemCache** | KV Pool 的 memcache 后端，官方口径**两大核心组件**（LocalService + MetaService，MemCache README）；ConfigStore 只是 MetaService 暴露的配置端点（6000，`ock.mmc.*.config_store_url`，kv_pool.md:550），不是并列第三层 | 提供 KV 存取；**layerwise 的唯一支持后端**（layerwise_kv_pool.md："requires the memcache backend"）——但依赖是单向的 layerwise→memcache：memcache 也可 non-layerwise 使用（kv_pool.md §3.5.1 官方示例 backend: memcache + use_layerwise: false）；`_BACKEND_CAPABILITIES` 的 `gva_layerwise` 是能力声明，不是排他限定 |
| **Hugepages** | Linux 大页内存 | memcache device transfer 硬性前置（`nr_hugepages=200000`，见 §3.7 探针第 0 项） |
| **Ascend 950 (A5)** | 下一代昇腾系列（950PR/950DT） | protocol 用 `device_urma`（UB）/ `device_uboe`（UBOE），与 A2/A3 不同（kv_pool.md:552） |

### 1.2 RoCE 机制一句话

把 IB（InfiniBand，原生 RDMA 专用网络）的报文头保留、装进以太网帧（v1 直接装以太网帧仅 L2 不可路由；v2 再包一层 UDP/IP 可路由，现网主流 v2），用 PFC（Priority Flow Control，优先级流控——让以太网在指定优先级上不丢包）保证无损——注意 RDMA RC 语义自带 go-back-N 重传，PFC 的价值是**避免触发昂贵重传**，不是"弥补没有重传"；剩下的一切（寻址、搬运、校验）全由网卡硬件完成，CPU **数据面绕过**（每次下发工作请求仍需 CPU 参与，并非只在注册内存时参与一次）。

### 1.3 MemFabric 搬运方向缩写

（D=Device 设备/HBM，H=Host 主机，R=Remote 远端）

- **D2D**：卡↔卡（同机或跨机）
- **D2H**：设备→主机　**H2D**：主机→设备
- **D2RH**：设备→远端主机　**RH2D**：远端主机→设备　**RH2H**：远端主机→主机（跨机）

### 1.4 protocol / 链路选型速查

**protocol 取值矩阵**（官方 kv_pool.md:552）——两个协议**都是设备侧直接搬、不经 host CPU**，区别只在"走哪条通道"：**HCCS 是板内总线（同机内），RoCE 是网络（可跨机）**。

| 取值 | 含义 | 怎么工作 | 什么时候选 |
|------|------|---------|-----------|
| `device_sdma` | SDMA over device | 设备侧 **SDMA 引擎**直接搬运 KV 数据，芯片之间走 **HCCS**（片间高速互连总线），同机内完成 | **仅 A3**（8卡16 die，HCCS 可用）——**A2 虽有 HCCS 硬件，但官方不支持 A2 走 device_sdma**；51 实况 |
| `device_rdma` | RDMA over device | 设备侧 **RDMA 语义**，数据经 NPU 的 **device RoCE 网卡**直接收发，不过 host CPU/内存 | 互连走 **RoCE 网络**的机器：**A2**（机内即 RoCE，推荐）；A3 配成 RoCE 链路也行；**跨机/走网络**搬 KV cache |
| `device_urma` / `device_uboe` | — | Ascend 950（A5）的 UB / UBOE 场景 | A5 才用 |

**怎么选（判别口诀）**：看机器互连是"板内总线"还是"网络"——
- A3 片间 = HCCS → **device_sdma**（仅 A3）
- 走 RoCE 网络（A2；或 A3+LINK_TYPE=ROCE）→ **device_rdma**
- 一句话：**A3/HCCS 机器用 sdma；A2 或 RoCE 链路用 rdma（A2 同机也是 rdma）**——A2 机内即 RoCE（MemFabric 官方："A2: DRAM+HBM pooling over Device RoCE"；kv_pool.md:1147 A2 直传 required `HCCL_INTRA_ROCE_ENABLE=1`）；两者都不经 host CPU。

**链路分支环境变量**（官方 env 脚本，kv_pool.md:188–233）：

```
A2（800I/800T A2）                     → 走 RoCE：hugepages + HCCL_IF_IP + GLOO_SOCKET_IFNAME + HCCL_INTRA_ROCE_ENABLE=1
A3 + LINK_TYPE=ROCE（有 device RoCE）  → 同上（protocol=device_rdma）
A3 + LINK_TYPE=HCCS（推荐）            → ACL_OP_INIT_MODE=1 + ASCEND_ENABLE_USE_FABRIC_MEM=1（protocol=device_sdma，51 实况）
```

**选型原则**：protocol 按"机器互连是 RoCE 还是 HCCS"二选一（KV 搬运的物理通道），别把 A3 的 `device_sdma` 照抄到 A2/RoCE 机器；完整落地见 §3.2。

---

## 2. mooncake master 拉起规范

> 适用：KV backend 选 mooncake 时（memcache 后端不需要 mooncake master，但有自己的 MetaService 进程，见第 3 章）。
> 参考脚本：`map_51/pr14465_dsv4_kvpool/start/`（历史位置 map_51/start/ 亦有）。

### 2.1 端口规划（硬规则）

| 用途 | 端口 | 原因 |
|------|------|------|
| master RPC | **50088** | 50051 被宿主机 OceanStor DTMA 占用（51 和 112 都踩过），**禁止用 50051** |
| metrics (HTTP) | **9008** | `curl http://127.0.0.1:9008/metrics`。⚠️ **9008 是本库脚本 `--metrics_port` 的覆盖值**，官方 mooncake 默认 **9003**（kv_pool.md §2.2.3）。换服务器时若按官方默认起 master，照抄 9008 必失败——以实际启动参数为准，二选一对齐 |
| vllm API | 8004 | 与 master/metrics 错开 |

启动前先做端口预检（ss/netstat 确认 50088/9008 空闲）。

### 2.2 mooncake.json 配置模板（实测有效）

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "ascend",
  "master_server_address": "127.0.0.1:50088",
  "global_segment_size": "5GB",
  "local_buffer_size": "5GB",
  "preferred_segment": false,
  "prefer_alloc_in_same_node": true
}
```

要点：
- **每卡 DRAM 配 5GB** 用于 pooling（global segment 5GB）
- ⚠️ **`local_buffer_size` 在 A3 fabric mem 下不生效**：开 `ASCEND_ENABLE_USE_FABRIC_MEM=1` 时，代码（mooncake_backend.py fabric 分支）硬编码 `local_buffer_size=0` 传给 store，官方 FAQ 5.3.2.2 同口径。只有非 fabric mem（A2/ROCE）分支才传递该值。→ A3 每卡实际贡献 = `global_segment_size` 一项，51 的 "local 5GB" 从未生效
- `protocol=ascend`（NPU 场景），metadata 用 P2PHANDSHAKE
- `prefer_alloc_in_same_node=true`：单节点验证默认值。注意改成 false 也**不会**让数据落到远端（见 §8.3，单节点失败路径不可构造的原因）

### 2.3 启动规范（四条纪律）

1. **日志必须重定向**到 `/home/lizhongyang/map_XX/mooncake_logs/`（master.log），禁止依赖终端回显
2. **PID 必须落盘**（master.pid），kill 时用
3. 启动后**探活四查**（全部通过才算 master 就绪）：
   - `role=leader`
   - `state=serving`
   - `service_ready=true`
   - TCP 50088 可连通
4. 等待 vllm worker 注册：`master_active_clients` 应等于 **DP × TP**（如 4×4=16），全部注册完再发请求

探活参考：`probe_mooncake_v2.sh` 模式（role/state/service_ready/TCP 四查）+ master.log `Master Admin Metrics` 行。

**可选调优参数**（官方 kv_pool §2.2.2；51 用默认未传，需要时按官方语义加）：
- `--default_kv_lease_ttl <ms>`：KV 对象租约 TTL，**必须大于 `ASCEND_CONNECT_TIMEOUT` / `ASCEND_TRANSFER_TIMEOUT`**，否则 get 阶段可能 `LEASE_EXPIRED`
- `--eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1`：达到高水位后按比例淘汰
- `--client_ttl <s>`：客户端存活 TTL（默认 10）；`enable_cpu_binding` 场景下 Ping 线程可能被 CPU 迁移错过心跳 → 建议抬到 60–120

**故障关键字速查**（官方 FAQ 5.3.1，出现即定位方向）：
- `NO_AVAILABLE_HANDLE` / `BatchPut failed ... insufficient space` → 淘汰后剩余空间放不下一次 BatchPut：加大容量 / 提高 eviction 余量 / 减小 batch
- `lease_expired_before_data_transfer_completed` / `LEASE_EXPIRED` → get 传输未完成租约已过期：加大 `--default_kv_lease_ttl`
- 既非 put 也非 get 的传输层错误 → 大概率 HIXL (ascend_direct) 问题，收集 `/root/ascend/log/debug/plog`

### 2.4 停止规范

- **显式 PID `kill -9`**（读 master.pid），**禁止依赖 `pkill -f mooncake_master`**——51 上实测杀不死（PID 19863/19865 残留）
- 重启前确认端口已释放，否则新 master 起不来

### 2.5 memcache 后端差异（指向第 3 章）

memcache 后端**有独立后端进程 MetaService**（等价 mooncake master，需单独拉起 + 探活 5000/6000/8000，官方 kv_pool §3.4），并有**自己的 Prometheus `/metrics` 端点（:8000，2026-08-31 实测确认）**，但指标集与 mooncake 的 master_* 完全不同；配置体系、环境变量、layerwise 额外启动参数亦不同——完整规范见第 3 章（**layerwise 验证必读**）。

### 2.6 常用检查命令

```bash
# master 核心指标（存证据用）
curl -s http://127.0.0.1:9008/metrics | grep -E '^master_(allocated_bytes|key_count|active_clients) '

# master.log 状态行（含 batch 速率）
grep "Master Admin Metrics" /home/lizhongyang/map_XX/mooncake_logs/master.log | tail -3

# master 是否活着 + 端口
ss -lntp | grep 50088

# 故障关键字定位（官方 FAQ 5.3.1）
grep -E 'NO_AVAILABLE_HANDLE|LEASE_EXPIRED|BatchPut failed' /home/lizhongyang/map_XX/mooncake_logs/master.log | tail
```

---

## 3. memcache 后端拉起规范

> 适用：KV backend 选 memcache 时。**memcache 与 layerwise 验证强相关**——PR14465 第 4 组
> （memcache layerwise）是 DSV4 NSA 硬限根因分析最详的一组，layerwise 验证首选 memcache 路径。
> 参考脚本：`map_51/pr14465_dsv4_kvpool/archive/layerwise_memcache_verify/`
> （start_layerwise_memcache.sh / update_mmc_conf.sh / probe_layerwise_memcache_env.sh）。

### 3.1 与 mooncake 的架构差异（决定拉起方式不同）

| 项 | mooncake | memcache |
|----|----------|----------|
| 独立后端进程 | 有（master，50088/9008，见第 2 章） | **有**（MetaService，等价 mooncake master，**需单独拉起 + 探活 5000/6000/8000**，见 §3.1.1） |
| 配置文件 | mooncake.json（脚本生成） | 容器内 site-packages 的 `mmc-local.conf`（**需手工改**，见 §3.2）+ `mmc-meta.conf`（MetaService 用，见 §3.1.1） |
| 环境注入 | MOONCAKE_* 变量 | `MMC_LOCAL_CONFIG_PATH` + source memcache/memfabric 的 set_env.sh |
| 指标观测 | `curl :9008/metrics`（allocated/key_count） | **有**：`curl :8000/metrics`（Prometheus，`memcache_allocated_bytes`/`memcache_alloc_requests_total` 等，**2026-08-31 实测确认可用**，见 §3.6） |
| lookup_rpc_port | `"1"` | `"0"` |

共同点：`--no-enable-prefix-caching`、长 prompt 门槛、存/取/去重判定逻辑（pool_scheduler/pool_worker 核心代码后端无关）。

#### 3.1.1 MetaService 独立启动（必须，v2 修订：memcache 不是"无 master 进程"）

官方（kv_pool §3.4）明确：**MetaService 是独立进程，只需在单个节点拉起**，等价 mooncake master。
不拉起时 vllm init 连 config_store(6000) 报 `errno:111`（51 腿3 实测根因，曾因 1b 旧版"无独立进程"误述导致该步被忽略）。

**① 启动命令（官方写法，优先）**：

```bash
export MMC_META_CONFIG_PATH={INSTALL_PATH}/memcache_hybrid/config/mmc-meta.conf
python -c "from memcache_hybrid import MetaService; MetaService.main()"
```

- `{INSTALL_PATH}` = `pip show memcache_hybrid` 的 Location
- **51 实测变体**：start_mmc_meta.sh / leg3 ensure_mmc_meta 均**未导出 MMC_META_CONFIG_PATH** 也能跑通（依赖安装路径默认配置）。可用，但新服务器建议用官方显式写法，避免装到非默认路径时读错 conf
- 启动脚本应幂等：vllm 拉起前若 5000/6000 已监听则复用（leg3 ensure_mmc_meta 模式），日志/PID 落盘

**② mmc-meta.conf 关键项**（与 mmc-local.conf 的一致性约束，kv_pool §3.3 Key Focuses）：

| 项 | 要求 |
|----|------|
| `ock.mmc.meta_service_url` | P/D 节点必须配**同一** MetaService 端点 |
| `ock.mmc.local_service.config_store_url` | 必须等于 mmc-meta.conf 的 `ock.mmc.meta_service.config_store_url`；两文件不一致 → init 失败 |
| `ock.mmc.meta_service.metrics_url` | `http://xx:8000`（官方配置项；**2026-08-31 实测确认可用**，`curl :8000/metrics` 返回 Prometheus 指标，见 §3.6） |
| `ock.mmc.local_service.world_size` | LocalService 上限（示例 256），含未来扩容 |

**③ 探活**：确认 5000（meta_service）/ 6000（config_store）/ 8000（metrics）TCP 可连通，并 HTTP 确认 `curl http://127.0.0.1:8000/metrics` 返回 Prometheus 文本，参考 start_mmc_meta.sh：

```python
python - <<'PY'
import socket
for p in (5000,6000,8000):
    s=socket.socket(); s.settimeout(1.5)
    try: s.connect(('127.0.0.1',p)); print(f'{p}: OPEN'); s.close()
    except Exception: print(f'{p}: CLOSED')
PY
```

### 3.2 mmc-local.conf 配置（启动前必做）

配置路径（py 版本不同会变，先 `find /usr/local -maxdepth 8 -name mmc-local.conf` 定位）：
```
/usr/local/python3.11.10/lib/python3.11/site-packages/memcache_hybrid/config/mmc-local.conf
```

必改两项：

| 配置项 | 值 | 原因 |
|--------|----|------|
| `ock.mmc.local_service.protocol` | **`device_sdma`**（51 A3 实测跑通；官方默认 host_rdma，**必须改**） | 官方（kv_pool.md:552）protocol 取值矩阵：`device_sdma`=SDMA over device，**推荐 A3**（HCCS 可用时）；**`device_rdma`**=RDMA over device，A2 与 A3（有 device RoCE 时）均支持、**推荐 A2**（跨机/RDMA 场景用这个）；Ascend 950 另设 `device_urma`（UB）/`device_uboe`（UBOE）；其余协议见 memcache 官方 mmc-local.conf。**选型 = 按机器互连（HCCS vs RoCE）二选一**，别把 A3 的 device_sdma 照抄到 A2/RoCE 机器 |
| `ock.mmc.local_service.dram.size` | **1GB 的整数倍**（51 实测 1GB 跑通） | device_sdma 模式要求对齐；⚠️ 官方另有口径：**A3 开 HCCS 时可设 0GB**（kv_pool §3.3 Key Focuses，fabric mem 场景）——两口径并存，实际取值按环境记入 env.txt |

操作纪律（update_mmc_conf.sh 模式）：
1. **先备份** `mmc-local.conf.bak.<timestamp>`
2. sed 替换后 grep 回显确认
3. 幂等（已是 device_sdma 不重复改）

### 3.3 环境变量（memcache 特有）

```bash
export MMC_LOCAL_CONFIG_PATH=<上节的 conf 绝对路径>   # 必须
export LD_LIBRARY_PATH={INSTALL_PATH}/memcache_hybrid/lib:${LD_LIBRARY_PATH}  # 官方示例显式 export（kv_pool §3.5）
# set_env.sh（存在则 source，缺了只 WARN 不阻断）：
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
```

性能相关（start_layerwise_memcache.sh 实测配置，**注意硬件分支**，按需取舍）：
```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export TASK_QUEUE_ENABLE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_INTRA_ROCE_ENABLE=1      # 对应 protocol=device_rdma 的 A2 分支（官方 required）；A3-HCCS 官方只要 FABRIC_MEM 一组（见下行），51 在 A3 上也设了（无害但勿照抄为 A3 必须）
export ASCEND_BUFFER_POOL=4:8
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
# A3-HCCS 官方分支（kv_pool §5.1）：ACL_OP_INIT_MODE=1 + ASCEND_ENABLE_USE_FABRIC_MEM=1（措辞 Recommended）
```

### 3.4 kv-transfer-config（backend 切换）

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "lookup_rpc_port": "0",
    "backend": "memcache",
    "use_layerwise": true,
    "layerwise_prefetch_layers": 3
  }
}
```

与 mooncake 的差异就两处：`backend=memcache`、`lookup_rpc_port="0"`。
layerwise 验证时 `use_layerwise=true` + `layerwise_prefetch_layers`（预取层数，实测 3）。

**端口命名注明**：官方内部自相矛盾——kv_pool.md 用 `lookup_rpc_port`，layerwise_kv_pool.md 用
`mooncake_rpc_port`。代码（pool_scheduler.py:978-984）两者都收但**推荐 `lookup_rpc_port`**，且对
`mooncake_rpc_port` 打 deprecation 警告（"will be removed in the future"）。**一律用 `lookup_rpc_port`**；
读到官方 layerwise 文档的 `mooncake_rpc_port` 时知道它是将废弃别名。

**layerwise 参数面（M4）**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `use_layerwise` | false | 仅 memcache backend 支持 |
| `layerwise_prefetch_layers` | 1 | 预取层数，提高传输/计算重叠，典型 1–4（实测 3） |
| `layerwise_max_transfer_blocks` | 0（不限） | 单批 transfer 最大块数，防单大层垄断总线 |
| `layerwise_max_transfer_bytes` | 0（不限） | 单批 transfer 最大字节数 |
| `h2d_stagger_us` | 0 | 多 TP rank H2D 拷贝错峰（如 TP8 设 100），缓解总线争用 |
| `discard_partial_chunks` | ⚠️ 见下 | 是否丢弃不完整 chunk 边界 |

⚠️ **`discard_partial_chunks` 文档-代码分歧**：官方 layerwise 文档称"layerwise 默认 false（保留部分层）"，
但本分支代码 pool_scheduler.py:136-137（`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/`）
**无条件默认 True**，全 ascend_store 目录未见 layerwise 分支改写。→ 验证"layerwise 保留部分层"语义时
**必须显式设 `discard_partial_chunks: false`**，不能依赖文档默认值。

### 3.5 layerwise 场景的额外启动参数（DSV4 实测集）

第 4 组实际使用的 layerwise 启动参数（相对 non-layerwise 的增量）：

```
--enable-expert-parallel
--disable-hybrid-kv-cache-manager      # ⚠️ 见下方警告
--tokenizer-mode deepseek_v4
--tool-call-parser deepseek_v4 --enable-auto-tool-choice
--reasoning-parser deepseek_v4
--quantization ascend
--block-size 128
--model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}'
--async-scheduling
--additional-config '{"ascend_compilation_config": {...}, "enable_cpu_binding": true, ...}'
```

**⚠️ 关键警告**：`--disable-hybrid-kv-cache-manager` **不能**绕过 DSV4 NSA multi-spec 硬限
（它是 vllm 主干的 KV group 调度层开关，与 vllm-ascend 的 `build_layerwise_reuse_layout`
校验层独立，实测仍抛 `Physical layer 2 with multiple cache specs...`）。
加这个参数的原因是 DSV4 hybrid attention 本身的调度需要，不是 layerwise 逃生门。
硬限详情见 §8.1。

另注意 `--enforce-eager` 与 `--compilation-config cudagraph_mode` 语义冲突
（enforce-eager 会覆盖 cudagraph 配置），二选一。

### 3.6 指标观测差异（判定方法必须调整）

memcache **没有** mooncake 的 master_* 指标，但 **MetaService 自带 Prometheus `/metrics` 端点（:8000）**，
2026-08-31 实测确认可用。三维证据链的观测方式：

| 维度 | mooncake | memcache |
|------|----------|----------|
| 存 | `master_allocated_bytes`/`key_count` 增长 | `curl :8000/metrics` 的 `memcache_alloc_requests_total` / `memcache_batch_alloc_requests_total`（counter，看增量）；`memcache_allocated_bytes` 是瞬时 gauge（pool 释放后可为 0，看趋势不看瞬间） |
| 取 | `External prefix cache hit rate` | 同左（vllm 侧指标，后端无关，**依然可用**） |
| 去重 | 重发后 key_count 恒定 | `memcache_alloc_requests_total` 增量观察 + server.log 侧间接 |

**C2 实测裁决（2026-08-31，51 宿主机）**：`curl http://127.0.0.1:8000/metrics` 返回合法 Prometheus 文本
（root `/` 404，仅 `/metrics` 有效），实际输出示例：

```
memcache_total_capacity_bytes{medium="hbm"} 0
memcache_total_capacity_bytes{medium="dram"} 0
memcache_allocated_bytes{medium="hbm"} 0
memcache_allocated_bytes{medium="dram"} 0
memcache_alloc_requests_total 696
memcache_alloc_successes_total 696
memcache_alloc_failures_total 0
memcache_batch_alloc_requests_total 12
memcache_batch_alloc_successes_total 12
```

→ 旧版 1b 断言"memcache 无等价 HTTP 端点"**被实测推翻（C2 方向为"可用"）**；
**存维可用 :8000/metrics 做独立证据源**，不再只能靠"代码路径等价性"论证。

PR14465 第 3 组（memcache non-layerwise）的判定方法：
`use_layerwise=false` 时与已 PASS 的第 1 组**共享同一代码路径**（pool_scheduler/pool_worker 统一实现，
仅 backend 存储实现不同）→ 服务就绪 + 推理正常 + External hit rate 即可判等效 PASS。
**不能**把 mooncake 的 allocated_bytes 标准生搬到 memcache。

### 3.7 启动前探针（probe_layerwise_memcache_env.sh 模式）

memcache 启动失败多为环境缺件，启动前按序核查：

0. **hugepages（M2，layerwise/memcache device transfer 硬性前置，缺了必须 abort 不只是 WARN）**：
   ```bash
   echo 200000 > /proc/sys/vm/nr_hugepages   # 官方 layerwise Prerequisites；持久化见 /etc/sysctl.conf
   grep -E 'HugePages_Total|HugePages_Free' /proc/meminfo   # 确认已分配
   ```
   112 PLAN P11 教训：缺 hugepages 时 memcache device transfer 不可用 → store 层 `batch_copy` 返回 -1、
   请求 0/20（51 腿3 失败主因假设）。**探测脚本中缺失即 FAIL/abort**，不是 WARN。
1. `pip show memcache_hybrid`（包存在 + Location）
2. `find /usr/local -maxdepth 8 -name mmc-local.conf`（配置在）+ protocol 字段按机器分支正确（A3/HCCS→`device_sdma`；A2/device RoCE→`device_rdma`，见 §3.2）
3. `find /usr/local -maxdepth 6 -name set_env.sh | grep -E "memcache|memfabric"`（环境脚本在）
4. **MetaService 存活**（§3.1.1 ③，5000/6000/8000 探活 + `curl :8000/metrics` HTTP 确认；缺进程时 vllm init 连 6000 报 errno:111）
5. `grep -rn "backend.*memcache" $VA/vllm_ascend/`（该代码版本支持 memcache backend）
6. `grep -rn "use_layerwise\|layerwise_prefetch_layers" $VA/vllm_ascend/`（layerwise 支持面）
7. 权重 config.json 关键字段（architectures / o_groups / quantization_config）
8. 端口与 NPU 现状（ss + npu-smi）

### 3.8 失败模式速查（memcache 特有）

| 现象 | 含义 |
|------|------|
| `Store initialization failed` / `memcache_backend.py assert res == 0` | mmc-local.conf 配错（protocol 未按机器分支设置：A3→`device_sdma` / A2→`device_rdma`，或 dram 未对齐）或 set_env 未 source |
| init 连 127.0.0.1:6000 报 `errno:111` | **MetaService 未启动**（§3.1.1），vllm 拉起前先拉起并探活 |
| `Physical layer N with multiple cache specs...` | DSV4 NSA 硬限（见 §3.5 警告），判 EXPECTED_FAIL 非 memcache 问题 |
| `Layerwise ... save batch_copy failed with return code -1` | store 层 transfer 失败，主因假设 = **环境缺件**（hugepages 未设 → device transfer 不可用）；先查 §3.7 探针 0 项 |
| 启动即退 `Configuration loading failed` | MMC_LOCAL_CONFIG_PATH 未导出或路径错（py 版本目录变化） |

---

## 4. vllm 池化场景启动

> 池化验证的 vllm 启动与普通推理不同：必须禁用本地 prefix caching、配 kv-transfer-config、
> 等待时间远超常规。参考实现：`map_51/pr14465_dsv4_kvpool/start/dsv4_mooncake_non_layerwise_start.sh`
> （通用 starter，KV_BACKEND × USE_LAYERWISE 环境变量切换）。

### 4.1 关键启动参数

```
--no-enable-prefix-caching          # 必须！否则本地命中抑制 KV pool 存取（见 §4.1.1）
--kv-transfer-config '{"kv_connector":"AscendStoreConnector","kv_role":"kv_both",
   "kv_load_failure_policy":"recompute",
   "kv_connector_extra_config":{"backend":"mooncake","lookup_rpc_port":"1","use_layerwise":false}}'
```

`kv_connector_extra_config` 三要素：

| 键 | mooncake | memcache |
|----|----------|----------|
| `backend` | `mooncake` | `memcache` |
| `lookup_rpc_port` | `"1"` | `"0"` |
| `use_layerwise` | 按验证矩阵 | 按验证矩阵（layerwise 可加 `layerwise_prefetch_layers: 3`） |

⚠️ **`kv_load_failure_policy=recompute` 有模型限制（M5）**：官方 kv_pool §1 明确 **hybrid attention
模型不支持 recompute**（DeepSeekV4、Qwen 3.5 等），且 vllm 默认值是 `fail`。→ hybrid 模型必须用默认
`fail`（或显式设）；Qwen3（GQA 非 hybrid）、DSV2-Lite 等才可用 recompute。DSV4 腿注意此约束。

其他常规参数：`--enforce-eager --trust-remote-code`、TP/DP 按空闲 chip 数、
`--max-model-len 32768 --max-num-batched-tokens 16384 --max-num-seqs 20 --gpu-memory-utilization 0.9`。

#### 4.1.1 为什么必须 --no-enable-prefix-caching

开启 prefix caching 时，重复前缀本地命中，connector 认为无需外部存取 → KV pool 指标全 0，
**误判为池化未生效**。验证日志中 `Prefix cache hit rate: 0.0%` 是禁用生效的确认（本身也是一条证据）。

### 4.2 关键环境变量

```
PYTHONHASHSEED=0                 # 必须！官方 required：跨节点统一 hash 生成（kv_pool §1；memcache/layerwise 同样需要，layerwise Prerequisites 显式列出）
VLLM_USE_V1=1                    # 强制 v1 引擎
VLLM_ENGINE_READY_TIMEOUT_S=2400 # 大权重加载 20-40 分钟
HCCL_BUFFSIZE=1024
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# A3 专用：
ACL_OP_INIT_MODE=1
ASCEND_ENABLE_USE_FABRIC_MEM=1
```

- `ASCEND_RT_VISIBLE_DEVICES` 用**逗号分隔**（`0,1,2,...`），不能用 `0..15` 范围写法
- mooncake 后端：导出 `MOONCAKE_CONFIG_PATH`（指向 mooncake.json）、`MOONCAKE_MASTER=127.0.0.1:50088`
- memcache 后端：导出 `MMC_LOCAL_CONFIG_PATH`、source memcache/memfabric 的 set_env.sh、
  mmc-local.conf 需提前改 protocol（详见第 3 章；layerwise 场景的额外启动参数也在 §3.5）

### 4.3 启动脚本模板要点（照抄 dsv4 starter 模式）

每个验证场景的启动脚本应包含（该模式在 4 组矩阵验证中稳定使用）：

1. **结果目录**：`results/<YYYYMMDD_HHMMSS>/` + `latest` 软链，内含：
   - `server.log`（完整日志，重定向）
   - `server.pid`、`status.txt`（RUNNING → PASS/FAIL/BLOCKED）
   - `env.txt`（**版本快照**：vllm/vllm-ascend pip version + location、git commit、npu-smi、全部关键环境变量——失败回溯的第一现场）
   - `command.sh`（实际命令行 printf %q 记录）
   - `mooncake.json`（mooncake 后端时，脚本生成）
2. **前置检查**：模型 config.json 存在、toolkit/ATB env 存在、端口空闲、可见设备数 ≥ TP
3. **启动**：后台 + 日志重定向 + PID 落盘
4. **READY 轮询**：每 10s 查 `/v1/models`，超时 2400s；进程死掉时 tail 日志并按关键字分类
   （`layerwise_cache_layout` → BLOCKED「layerwise 不支持该模型」而非 FAIL，见 §8.1）

### 4.4 启动失败速查（池化特有）

| 现象 | 含义 |
|------|------|
| `RuntimeError: Worker failed ... layerwise_cache_layout` | 模型触发 multi-spec 硬限，见 §8.1 |
| `Initialize mooncake failed` / `Connection refused` | master 未起 / 端口配错（查 50088） |
| 主 vllm 活着但 8004 不监听、Worker 卡 do_poll | EngineCore 已崩但未发终止信号 → 手动 kill 主 PID + `pkill -9 -f VLLM::`，看 server.log 找崩溃堆栈 |
| `Failed to infer device type` | NPUPlatform 插件加载失败，查 plugin 加载阶段 |
| KV cache spec 校验阶段 HBM 只到 baseline（~18GB/die）就停 | shard loading 未完成即被 spec 校验挡下 |

### 4.5 就绪后第一件事

`/v1/models` 通了 ≠ 池化就绪。确认：
1. master.log 出现 `role=leader, state=serving, service_ready=true`
2. `master_active_clients` = DP×TP（全部 worker 注册完成）
3. server.log 中 N 个 `AscendStoreConnector` 创建成功（N = DP×TP，DSV4 4×4=8 connector/4DP 实测）

然后才进入测试（见第 5 章）。

---

## 5. 池化测试方法

> 如何设计测试请求、如何组织验证矩阵、如何做对照实验。
> 参考脚本：`map_51/pr14465_dsv4_kvpool/test/`（auto_verify_clean.sh / verify_load_new.sh / multi_growth_requests.sh）。

### 5.1 长 prompt 门槛（最常踩的坑，先算再发）

**存入条件**：`num_tokens_to_save ≥ chunk_boundary`，其中
`cache_transfer_granularity = lcm_block_size`（DSV4 = **4096**）。

| 模型/场景 | granularity | prompt 要求 |
|-----------|-------------|-------------|
| DSV4-Flash（PR14465） | 4096 | **≥ 2 个完整 4096 块 → ≥ 8192 token** |
| PR14912 验证场景 | 128 | ≥ 5042 token（实测门槛） |

规则：**先确认目标模型的 cache_transfer_granularity，prompt ≥ 2 个完整块**。
短 prompt（如 629 token）→ `num_tokens_to_save=0` → `skip_save=True` → 整个请求不存入，
master 指标全 0，极易误判为"池化未生效"。

实用做法：基础段落重复拼接。参考 auto_verify_clean.sh——
~182 token 的英文段重复 260 次 ≈ 11.8K token（`num_tokens_to_save=8192 > 4096`），稳定触发存入。

**注意块对齐**：仅长度超过还不够，块 hash 需对齐（`available_full_block_count = len(block_hashes) // hashes_per_transfer_block`）。
PR14465 首跑 3 条 ~13K prompt 全 0 就是对齐问题，改用规整重复文本后成功。

### 5.2 验证矩阵设计（backend × layerwise）

池化 PR 验证的标准矩阵（PR14465 模式，4 组）：

| # | backend | use_layerwise | DSV4 预期 |
|---|---------|---------------|-----------|
| 1 | mooncake | false | PASS（核心路径） |
| 2 | mooncake | true | EXPECTED_FAIL（DSV4 NSA 硬限） |
| 3 | memcache | false | PASS |
| 4 | memcache | true | EXPECTED_FAIL（同根因） |

要点：
- **先跑 non-layerwise**（核心功能），PASS 后再碰 layerwise
- layerwise 失败若为 multi-spec 硬限 → 判 **EXPECTED_FAIL / BLOCKED**，不算 PR 缺陷（见 §8.1）
- 同一矩阵内**干净基线**：每组开始前全清重启（kill vllm + master → counters 归零），参考 `full_restart_clean.sh`

### 5.3 标准测试流程（存 → 取 → 去重）

```
0. （推荐）**warm-up 预热**：ASCEND_BUFFER_POOL 场景下 HCCL one-sided 连接懒建（官方 kv_pool §2.4.2，
   每连接 4MB device mem），压测前先发 2 个单请求预热（官方建议：input 8K / output 1 / 总请求数 2–3× 设备数）
   ——消除首次连接耗时与断连对指标/成功率的污染
1. 等 READY（status.txt 出现 PASS: service ready）
2. 记录基线 counters：
   curl -s http://127.0.0.1:9008/metrics | grep -E '^master_(allocated_bytes|key_count|active_clients) '
   → 干净基线应为 0 / 0 / DP×TP
3. 【存】并发发 N 个长 prompt（N=20，curl 后台并发），max_tokens 64 即可（省时间）
   → 请求完 + 15s 后再采样 counters，应增长
4. 【取+去重】用完全相同的内容重发若干次（verify_load_new.sh 模式）
   → key_count / allocated 应恒定不增长（去重）
   → server.log `External prefix cache hit rate` 应 > 0（取）
5. 【增长】发不同内容长 prompt ×3（multi_growth_requests.sh 模式）
   → keys 应阶梯增长（61 → 244 这类）
6. 全程采样 master.log 的 Batch Requests 速率行（PutStart/Get/ExistKey 的 Req/Item 每秒速率）
```

并发参考：轮次递增设计（3 → 50 → 30 → 30 不同内容 → 30 不同+主题段），
第一轮小并发先验证链路，再加压。

### 5.4 对照实验设计（控制变量定位根因）

当"存取不发生"时，用标准 attention 模型做对照（PR14465 的 Qwen3-8B 实验）：

- 相同 mooncake 配置，只换模型（DSV4 hybrid attention ↔ Qwen3 标准 attention）
- Qwen3 能存取 → 排除 `--no-enable-prefix-caching` / 权重 / 环境问题 → 根因在模型特性
- 实测结论：DSV4 早期全 0 的根因是 granularity=4096 + 短/未对齐 prompt 的 skip_save，不是 prefix-caching

**原则：怀疑某配置是根因时，固定其他变量单独翻转它**（用户偏好的控制变量比对法）。

### 5.5 layerwise 路径的额外观察点

layerwise（`use_layerwise=true` + `layerwise_prefetch_layers`）验证时额外看：

1. 启动期：`build_layerwise_reuse_layout` 是否抛 multi-spec ValueError（DSV4 必抛，见 §8.1）
2. 运行期：kv_transfer 日志的 layer load 事件时间戳（PR14912 的 `_TimedLayerLoadEvent` 语义）
3. 指标口径与 sync 不同（load_keys = blocks×layers，放大 64 倍，**跨 path 不可直接对比**，见 §8.4）
4. layerwise 存入时机在请求完成后（non-layerwise 亦然），不要期望请求中途看到 PutStart
5. **layerwise 激活面（配置证据）**：worker 日志 `layerwise config: num_layers=N num_groups=M`——没有这行说明 gate 没开（backend 未 opt-in / 参数没传到）
6. **layerwise 判据的正计数形态（PR15367 实测，memcache）**：
   - 存侧：`load_gvas: req=... keys=N valid_gvas=N lease_fail=0`（keys>0 即入池；lease_fail=0 排除租约失败假阳性）
   - 取侧：`hit_check: req=... token_len=N hits_per_group=[N] hit_tokens=N`（hit_tokens>0 即命中）
   - 注意首行 `hit_check ... token_len=15 no participating groups` 是预热小请求的正常跳过，判据用 `hit_tokens=` 的行（长前缀请求）

### 5.5b PD 分诊（kv_producer/kv_consumer + MultiConnector + proxy）观察点（PR15367 S1，单机 8 卡实测）

PD 分诊拓扑：proxy :9000 → P :8100（TP=4，MooncakeLayerwise + AscendStore/memcache layerwise 双 connector）→ D :8200（TP=4，MooncakeLayerwise consumer）。判据（全部正计数）：

| 判据 | 证据形态 | 封堵的失效 |
|------|---------|-----------|
| 成功率 100% | `success_rate=5/5`，每请求带回真实 `prompt_tokens=N completion=M`（死服务给不出带 token 计数的 completion） | 服务没起来 |
| connector 初始化无异常 | P/D 日志 `Creating v1 connector with name: AscendMultiConnector` ×N，grep `AttributeError` = 0 | 初始化路径崩（waiter 下沉类重构的关键回归点） |
| P 侧池写活跃 | prefill.log `load_gvas:` 行数 > 0 + MetaService `query_successes` > 0 | 起来了但没入池 |
| D 侧接收完整 | decode.log layerwise recv 行数 > 0 + **27 层 LayerMetadata（`model.layers.0`–`26`）完整经 metaserver 到达 P**，`remote_engine_id` 与 D 配置一致 | KV 传输链路断裂/静默丢层 |

要点：LayerMetadata 逐层数（0 到 num_layers-1 连续）是"传输内容完整"的强判据；P 侧发送行数与 D 侧接收行数应同量级（浮动属正常，判据是 >0）。

### 5.6 测试脚本纪律

- 复杂流程写 .sh push 到远端跑（PowerShell → ssh 引号转义不可靠，见 AGENTS.md §1）
- 长 prompt 不要硬编码在命令行里，脚本内拼接
- 每轮测试的输出重定向到日志文件并保留（验证过程日志必须留档）
- 请求体存 `/tmp/creq_$n.json` 便于事后检查响应内容是否正常
- **服务器到 GitHub 断连时的代码部署走 git bundle**：本机 `git bundle create f.bundle '分支名' '^基线'`（⚠️ PowerShell 会吞 `..` range 且裸 hash 不产生 ref 条目，必须引号分支名 + ^排除式）→ scp 宿主机 → docker cp 进容器 → `git fetch /tmp/f.bundle 'refs/heads/b:refs/heads/from-bundle'` → `git reset --hard <sha>`；fetch 失败一次先 timeout 60 重试再降级 bundle
- **e2e 多场景编排在宿主机 nohup 跑总脚本**（更新代码 → clean → 前置服务 → 逐场景 start/test/stop → clean），单场景失败不中断后续场景；场景间必须 stop + clean_npu，否则 HBM 残留污染下一场景

---

## 6. E2E 通过判定标准

> **什么样的结果才算 E2E PASS**。三维证据链缺一不可；判定前先扫第 7 章（误区清单）。
> 蓝本：PR14465 第 1 组验收表（map_51/pr14465_dsv4_kvpool/record_final/02）。

### 6.1 三维证据链（核心标准）

E2E PASS = 以下三维**同时**成立，且每维都有留档证据：

#### ① 存（Put）
| 证据 | 通过条件 |
|------|----------|
| `master_allocated_bytes` | 干净基线 0 → **> 0**（如 41.31 MB） |
| `master_key_count` | 0 → **> 0**（如 61；不同内容多轮可至 244） |
| master.log batch 指标 | `PutStart:(Req=.., Item=..)` 非零速率 |
| （⚠️ `master_put_start_requests_total` 恒 0 是**正常**的，见 §7.1） |

#### ② 取（Get/ExistKey）
| 证据 | 通过条件 |
|------|----------|
| `External prefix cache hit rate` | **> 0**（首跑 37-42%，复验 9.0%，量级取决于负载） |
| `Prefix cache hit rate`（本地） | **= 0.0%**（证明确实禁用了本地缓存，命中来自外部 pool） |
| master.log | `Get:(Req=..)` / `ExistKey:(Req=..)` 非零速率 |

#### ③ 去重（不重复写入）
| 证据 | 通过条件 |
|------|----------|
| 相同内容重发 ×N | `key_count` / `allocated_bytes` **恒定不增长** |

三维关系：② 依赖 ①（先存才能取）；③ 证明 key 语义正确（同内容同 key）。
只做 ① 不做 ②③ = 半程验证，不能判 PASS。

#### ④ memcache 后端的等价三维口径（PR15367 实测，2026-09-01）

mooncake 的 master_* 指标在 memcache 场景换为 MetaService 证据（:8000/metrics 拉取）+ vllm 侧指标：

| 维度 | mooncake 口径 | memcache 等价口径（实测形态） |
|------|--------------|------------------------------|
| 存 | `master_allocated_bytes` / `master_key_count` 增长 | MetaService `alloc_successes=N stored_keys=N`（另一进程的独立计数）+ worker 日志 `load_gvas: keys=N valid_gvas=N` |
| 取 | `External prefix cache hit rate > 0` | vllm `/metrics` 的 **`external_prefix_cache_hits_total`（counter，精确断言用这个而非滑动窗口 rate）** + scheduler 日志 `hit_check: hit_tokens=N` |
| 去重 | 重发后 key_count 恒定 | 重发后 `stored_keys` 不再增长（或按 key 语义判等） |

注意：memcache 1.2.0 无 put 类命名指标（memcache_*_alloc_* 是分配口径不是写入口径），存维以 MetaService `stored_keys` + `load_gvas` 日志双源交叉（见 §6.6 跨进程证人原则）。

### 6.2 服务健康（前置条件，不算 PASS 的一部分但必须确认）

| 项 | 标准 |
|----|------|
| master 状态 | `role=leader, state=serving, service_ready=true` |
| `master_active_clients` | = DP × TP（如 16），全部 worker 注册 |
| connector 创建 | server.log 中 N 个 `AscendStoreConnector`（N=DP×TP） |
| device type | `device_type=npu`（NPUPlatform 插件加载成功） |
| 推理本身 | 请求正常返回、内容正常（池化不能破坏推理） |

### 6.3 验收表模板（每个场景一张，随证据归档）

```markdown
| 维度 | 结果 | 证据 |
|------|------|------|
| 服务启动 | ✅/❌ | connector 数、master serving、active_clients |
| 本地 prefix cache 禁用 | ✅ | Prefix cache hit rate: 0.0% |
| KV pool 存入 | ✅ | allocated_bytes=xxx, key_count=xxx |
| KV pool 取 | ✅ | External prefix cache hit rate=xx% |
| 去重 | ✅ | 重发后 keys 恒定 |
| 推理正确性 | ✅ | 响应内容 spot check |
```

归档位置：`map_XX/<pr_task>/record_final/`（结论 + evidence/ 原始文本快照）。

### 6.4 FAIL / BLOCKED / EXPECTED_FAIL 的区分

| 判定 | 含义 | 典型例子 |
|------|------|----------|
| **PASS** | 三维证据链齐全 | 第 1/3 组 |
| **FAIL** | PR 代码缺陷导致 | 启动崩溃且根因在 PR 改动内 |
| **EXPECTED_FAIL / BLOCKED** | 已知环境/上游硬限，**非 PR 缺陷** | DSV4 NSA multi-spec 硬限（§8.1）、master 未配置 |

判定 EXPECTED_FAIL 必须给出根因代码位置 + 对照证据（同代码 non-layerwise PASS /
单 attention 模型 layerwise PASS），否则降级为 FAIL 处理。

### 6.5 干净基线原则

- 每组验证前**全清重启**（kill vllm + mooncake，counters 归零），否则增长无法归因
- 基线采样留档（请求前的 counters 快照）
- PR 分支代码更新后须**复验**：干净基线重跑核心组，对比关键值（如 keys=61 / 41.31MB 两次复验一致 → 行为等价结论）
- 版本快照（env.txt：commit + pip version + location）随每组留档，防"验证的到底是哪个 commit"争议

### 6.6 虚假通过防范（vacuous pass 防范，PR15367 沉淀）⚠️ 判据设计层，先于一切 PASS 判定

**问题**：e2e"通过"可能是假的——三类典型形态：**A 服务根本没起来**；**B 起来了但 KV 没入池**；**C 入池了但静默不命中**（无报错、全 miss，验证却"通过"）。C 在 key 格式重构/迁移类 PR 中最阴险：key 静默漂移 = 永远 miss + 零报错。

**总原则一句话**：判据全部是"正计数 + 跨进程证人"，静默失效的任何一类都会让对应计数归零、判据 FAIL，而不是侥幸通过；阴性对照排除"验证方法本身失效"。

**① 正计数判据（不是"无错误"判据）**

- PASS 条件必须写成**计数 > 0** 的形态（`hit_tokens=3328`、`valid_gvas=26`、`stored_keys=28`），不能写成"无 Traceback / 无 Segfault"
- 无错误判据对 C 类静默失效**天然失明**：全 miss 也是无错误

**② 跨进程证人原则：每类失效模式配一个独立于被测代码的证人**

| 失效模式 | 证人（被测代码之外） |
|---|---|
| A 服务没起 | 每请求带回真实 `prompt_tokens=N completion=M`——死服务给不出带 token 计数的 completion |
| B 没入池 | 存储后端**自己进程**的计数（mooncake `master_key_count` / memcache MetaService `stored_keys`）——worker 代码谎报存入不可能让服务端计数增长 |
| C 静默不命中 | vllm `/metrics` 的 `external_prefix_cache_hits_total`（框架聚合，与 scheduler 日志的 `hit_tokens` 跨层对拍） |

**③ 双向断言（证明测试对目标失效是敏感的，不是恒通过的）**

- 不仅断言"命中 > 0"，还要断言"miss 被真实记录"：MetaService `query_not_found=28`（首发请求 miss 数）
- 若 key 静默漂移：会计变成"恒 miss 无报错"→ `hit_tokens=0` → 判据 FAIL。**query_not_found 的存在证明 miss 通道也在被观测**，测试有区分度
- 类似：`lease_fail=0` 与 `valid_gvas>0` 并读，排除"租约失败导致假阳性"

**④ 跨层算术一致（多源对拍）**

同一物理量在不同会计路径应可对账：`load_gvas keys=26` × 128 token/块 = `hit_tokens=3328` = `/metrics external_prefix_cache_hits_total=3328.0`；`load_gvas` 与 `hit_check` 应是**同一请求 ID**（`req=chatcmpl-...`）。编造或读错文件很难做到三处算术自洽。

**⑤ 阴性对照（negative control）——排除验证方法本身失效**

- 设计一个"该指标必然为 0"的场景做对照组：如 mooncake 非 layerwise 场景（S3）之于 memcache layerwise 场景（S2）
- 用**同一组 grep/判定命令**跑两个场景：S2 `load_gvas:` 行数 >0 且 S3 = 0 行 → 证明 grep 模式有效、读的是本轮日志、能区分路径
- 若验证方法本身坏了（grep 错文件/模式恒不匹配/读旧缓存日志），S2 也应得 0。**S2>0 且 S3=0 的不对称才是方法有效的证明**——单看 S3=0 毫无意义（恒真），单看 S2>0 不能排除巧合

**⑥ 数值敏感性证明读的是活数据（不是死文件/常量）**

- 两轮验证的数值应**不同**（时序浮动本身是活数据证据）：hit_tokens 3456→3328、请求 ID 不同
- 日志行号漂移（`pool_scheduler.py:389`→`:388`）与两棵源码树的 rebase 偏移吻合——佐证两轮跑的各自声称的代码版本（配 §6.5 版本快照：容器内 `git rev-parse HEAD`）
- 若两轮输出完全一致，反而要怀疑读了归档的旧日志

### 6.7 重构类 PR 的双轮夹逼复验（behavior-preserving 验证）

重构（宣称行为不变）的 PR，"行为等价"不能靠单轮 PASS 推断，**必须双轮夹逼**：

- **轮 1（基线）**：重构前的 head 跑完整场景集，全部 PASS 才作为有效基线
- **轮 2（复测）**：重构后（含 rebase / 检视返工 / bugfix）的 head 再跑**同构**场景集
- 两轮同构 PASS = 重构未引入行为漂移的实证（不是推断）；中间夹着的所有高风险变更被夹逼验证
- 数值浮动按 §6.6⑥ 归因（滑动窗口命中块数、请求分布），判据是 >0 类正计数，非定值对拍
- 报告双轮并列（`e2e-report-*.md` ×2），互相引用；PR 描述 Test plan 写"re-verified on <head> + validated earlier on <基线> (two rounds, same results)"

### 6.8 判定流程图

```
请求发出且正常返回？
├─ 否 → 查启动/推理问题（§4.4），不算池化结论
└─ 是 → 存：allocated/key_count 增长？
        ├─ 否 → 查误区：skip_save（prompt 长度/对齐）？prefix-caching 没禁？
        └─ 是 → 取：External hit rate > 0？
                ├─ 否 → 查 master Get/ExistKey 速率、lookup ZMQ 链路
                └─ 是 → 去重：重发 keys 恒定？
                        ├─ 否 → key 生成语义问题，FAIL
                        └─ 是 → E2E PASS，归档三维证据
```

（判定前自检 §6.6：你的判据是正计数吗？有跨进程证人吗？阴性对照在哪？——三问答不上来的 PASS 是可疑的。）

---

## 7. 判定误区清单

> 判 PASS/FAIL 前必扫。来源：PR14465/PR14912 实测踩坑（详见各 record_final / HANDOVER）。

### 7.1 `master_put_start_requests_total = 0` ≠ 池化未生效 ⚠️ 最高频误区

- 该 Prometheus counter 只计**单条** Put API
- mooncake 后端走 **batch API**（`batch_put_from_multi_buffers`），其统计只在 master.log
  `Batch Requests` 字段以每秒速率呈现
- **正确判定**：看 `master_allocated_bytes` / `master_key_count` / master.log `PutStart:(Req=..)`

### 7.2 `External prefix cache hit rate` 非实时

- 是**最近 1000 请求的滑动窗口平均**，请求结束后不自更新
- 数值量级与负载强相关（首跑 37-42%，干净复验 9.0%，都算 PASS）——只判 > 0，不设阈值

### 7.3 短 prompt "全 0" 是 skip_save，不是 bug

- `num_tokens_to_save < chunk_boundary` → `skip_save=True` → 整请求不存入
- granularity 因模型而异（DSV4 4096 / PR14912 场景 128）
- 详见 §5.1；先算门槛再发请求

### 7.4 本地 prefix cache 不禁用 → 永远看不到池化效果

- 本地命中抑制 connector 外部存取
- 必须确认 server.log `Prefix cache hit rate: 0.0%`（这本身是一条证据）

### 7.5 Qwen3 对照实验的教训：别急着怪配置

- "DSV4 全 0"最初怀疑 prefix-caching/权重，Qwen3-8B 对照（相同配置能存取）一次排除三个假设
- **怀疑配置时先做控制变量对照**，再下结论

### 7.6 主 vllm 活着 ≠ 服务可用

- EngineCore 崩溃后 Worker 卡 do_poll、8004 不监听、status.txt 不生成，但主进程存活
- 判启动失败要看 server.log 崩溃堆栈 + 端口监听，不能只看进程

### 7.7 `pkill -f mooncake_master` 杀不死

- 51 实测 PID 残留（19863/19865）
- 一律显式 PID `kill -9`；重启前确认端口释放

### 7.8 指标聚合的 TP rank 放大（PR14912 A1）

- 框架聚合**全部 TP rank** 的 worker stats → `load_count` / `load_keys_total` 放大 TP 倍
- avg / P90 不受影响；看板速率类聚合需 ÷TP

### 7.9 跨 path 指标不可直接对比（PR14912 m3）

- layerwise 的 `load_keys` = 跨层 block 数（blocks × layers），比 sync 的 "rank chunk 数" 放大 64 倍
- sync / layerwise / async 三条路径的计数**口径不同**，各自纵向对比，不横向比

### 7.10 master 重启不影响已存数据（失败路径构造误区）

- mooncake 同节点池把 KV 存 worker 本地宿主段，master 重启只清元数据不清数据 → get 继续命中
- `prefer_alloc_in_same_node=false` 也一样（数据段仍落本地）→ **单节点无法构造 load 失败**（见 §8.3）

### 7.11 HBM 到 baseline 就停 = spec 校验阶段崩溃

- 加载中每 die ~18GB baseline，完成 ~50GB；停在中间说明被 KV cache spec 校验挡下（如 layerwise 硬限）
- 不是权重/磁盘问题

### 7.12 版本验证要用 pip location 而非记忆

- `pip show vllm vllm-ascend` 看 **Editable project location** 是否指向预期源码目录
- editable 残留（旧 .pth / dist-info / egg-info 在挂载卷上）会让"以为切了分支其实跑的旧代码"——卸载规范见 env_install/4

---

## 8. 已知硬限与口径限制

> 这些不是 PR 缺陷，是 vllm-ascend / 环境的已知边界。判定 EXPECTED_FAIL / 设计验证方案前必读。
> **新发现的硬限发现后必须回写本章**（与 env_install 回写规则一致）。

### 8.1 DSV4 NSA 多 cache spec × layerwise = 硬限（最重要）

**位置**：vllm-ascend `distributed/kv_transfer/kv_pool/ascend_store/layerwise_cache_layout.py`
（`build_layerwise_reuse_layout`，PR14465 时为 :215-228，后续版本行号可能漂移）

**规则**：每个物理层若有多个 cache spec，必须**恰好 1 个 main + 1 个 `.indexer.k_cache`**（SFA 模式），
否则 `raise ValueError`（有 UT 明确断言拒绝：`test_layerwise_cache_layout.py::test_ambiguous_multi_spec_layer_is_rejected`）。

**触发**：DSV4-Flash（NSA 架构）layer 2 起 self_attn 含 **5 个 cache spec**：

| spec | 分类 |
|------|------|
| `compressor.state_cache` | main |
| `indexer.k_cache` | indexer |
| `indexer.compressor.state_cache` | main |
| `swa_cache` | main |
| `attn` | main |

→ main=4 ≠ 1 → ValueError → EngineCore 崩（错误信息 `Physical layer 2 with multiple cache specs...`）。

**边界**：
- `--disable-hybrid-kv-cache-manager` **不能绕过**（hybrid_kv_cache_manager 是 vllm 主干的调度层，与 vllm-ascend 的 layerwise 布局层独立）
- non-layerwise **不受影响**（`cache_coordinator=None` 走直接路径，不调 `build_layerwise_reuse_layout`）——同权重 non-layerwise PASS 是标准对照证据
- **绕行方案（v2 修正，拆两层）**：
  - **验证 layerwise 传输**：必须用 **MLA/SFA 模型**（官方支持矩阵，示例 DeepSeek-V2-Lite）——full attention（`attention_v1`）**未集成 layerwise wait/save**，用它验证得到的是"布局不崩"而非"layerwise 传输工作"
  - **仅验证布局不崩 / 与 NSA 硬限解耦**：才用单 attention 模型（Qwen3 / Llama-3 / Qwen3-32B-pdmix 等），且结论只能写"**布局通过**"，不得写"layerwise 验证通过"

**代码实证（v2）**：`wait_for_kv_layer_from_connector` / `maybe_save_kv_layer_to_connector` 的调用面——
mla_v1(:1901/:2019)、sfa_v1(:1550/:1590/:1698)、dsa_v1(:1250/:1260/:1284)、dsa_cp(:1434/:1513) 均有调用；
**attention_v1 为 0 处**（官方 Supported Models 段亦明文，但其"CP 变体均未集成"表述滞后于代码——dsa_cp 已有调用，引用模型支持面以代码 grep 为准）。

**PR15307 Q2 最终裁决（2026-08-30 实测，E2E 腿 2）**：
- **结论：触发（维持硬限成立）**。DSV4 + layerwise 在 worker 层 `get_kv_cache_spec → _get_layerwise_kv_cache_memory_info → build_layerwise_reuse_layout`（vllm-ascend refactor_layerwise_B，commit `b3a141331`）抛出同一 `Physical layer 2 with multiple cache specs` ValueError，5 spec 集合与顺序与 PR14465 堆栈**逐项一致** → 证实 PR15307 GVA 线程收敛重构**未改动** spec 收集/分组逻辑，该硬限是**上游存量 feature gap，非 PR15307 缺陷**。
- **裁级**：DSV4 + layerwise = **EXPECTED_FAIL**（上游缺口），后续方向：升级 `build_layerwise_reuse_layout` 支持 NSA 多 main spec（需区分 attn/swa_cache/compressor.state_cache/indexer.compressor.state_cache 的 main 归属，或将 NSA 各子 cache 归并为一个聚合 spec）。
- 留档：`map_51/pr15307_gva_threads/evidence/leg2_dsv4_layerwise/`（error_stack.txt 完整调用链）。

**相关调用点**（评审 layerwise 改动时关注）：`pool_worker.py` / `pool_scheduler.py` / `worker.py` 各 1 处。

### 8.2 存入时机与 token 门槛（语义限制）

- non-layerwise 的 KV 存入发生在**请求完成后**（非流式中途）
- `num_tokens_to_save < chunk_boundary` 且无 partial block → `skip_save=True`（详见 §5.1）
- 这是**语义正确**的设计，不是丢失；验证用例必须跨过门槛

### 8.3 单节点无法构造 load 失败路径（PR14912 结论）

- mooncake 同节点池把 KV 存 **worker 本地宿主段**，master 重启只清元数据不清这段数据 → get 继续命中
- 改 `prefer_alloc_in_same_node=false` 无效：数据段仍落本地持久化段
- 触发真实 load 失败需要：**多节点部署**（远端节点数据丢失）或物理删 worker 本地数据段（共享服务器不可安全构造）
- **处置**：失败观测路径由单测覆盖（`_record_load_finished(..., num_failed_keys=1)` 断言），交付表述为"1 指标仅注册未真实触发，有单测保护"

### 8.4 指标口径差异（跨 path 不可比，PR14912）

| 口径 | 含义 | 注意 |
|------|------|------|
| sync `load_keys` | rank chunk 数 | 基准 |
| layerwise `load_keys` | blocks × layers | 放大 ~64 倍，**不可与 sync 直接比** |
| worker stats 聚合 | 全 TP rank 求和 | `load_count`/`load_keys_total` 放大 TP 倍，速率类看板需 ÷TP |
| `load_failed_keys_total` | 失败键 | 见 §8.3，单节点恒 0 属预期 |
| layerwise 计时 | 传输线程 set 事件时刻起算 | C 轮修复后语义（`_TimedLayerLoadEvent`） |

### 8.5 模型侧约束速查（51 服务器实测）

| 模型 | 池化验证适用性 |
|------|----------------|
| DeepSeek-V4-Flash(-DSpark-w4a8) | non-layerwise ✅ / layerwise ❌（§8.1 硬限）；granularity=4096，prompt ≥ 8192 |
| Qwen3-32B(-pdmix) | 对照/control 首选；PR14912 三路径全通（granularity=128 场景 prompt ≥ 5042） |
| Qwen3-8B | 标准注意力，快速对照实验用 |
| 单 attention 架构（Llama-3 类） | ⚠️ 仅"layerwise 布局验证 / 与 NSA 硬限解耦"，结论只能写布局通过（`attention_v1` 未集成 layerwise wait/save）；**layerwise 传输验证必须用 MLA/SFA 模型**（DeepSeek-V2-Lite 等，见 §8.1） |

### 8.6 环境侧硬约束（与 AGENTS.md §6 交叉）

| 约束 | 说明 |
|------|------|
| 50051 端口 | 宿主机 OceanStor DTMA 占用（51、112 均有），mooncake 一律 50088 |
| vllm ↔ vllm-ascend 版本配对 | 见 AGENTS.md §2 Step5 配对表（配错启动即崩） |
| 共享服务器 NPU | 先 npu-smi 查空闲，只申明空闲 chip |
| `master_put_start_requests_total` | 单条 API 计数，batch 后端恒 0，非故障（§7.1） |