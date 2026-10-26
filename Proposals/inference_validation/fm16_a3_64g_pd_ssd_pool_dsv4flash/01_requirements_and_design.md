
# 01 适用条件与验证方案

本套流程说明如何在 FM16 A3 64G 双机上拉起 DeepSeek-V4-Flash 的 PD 分离服务，接入 memcache 的 DRAM/SSD 池，并检查保存与命中是否发生。它从 [pool_64g_a3_profiling](../../ascend_project/pool_64g_a3_profiling) 提取实际使用的配置、启动入口和测试方法；原目录名中的 profiling 不属于本流程。

本文档集记录操作方法和必要条件，不收录历史测试成绩，不代表整理后的入口已经重新实测。源目录及其中的配置、日志、归档和其他验证材料全部保留。本目录只维护五份文档；配置和脚本全文见 [03](03_verification_scripts.md)，顺序操作见 [04](04_verification_env_and_launch.md)，测试观察与判定见 [05](05_verification_process_and_results.md)。

## 1. 适用环境

| 项目 | 本流程采用的条件 |
| --- | --- |
| 平台 | 沿用来源场景名称 FM16、A3、64G；P/D 各一台，来源记录每台 8 卡、16 die。卡与 die 的内存口径在执行前用设备清单核对，不能由目录名推算。 |
| P 节点 | 80.48.29.133，容器 `wl_1011`，DP2 × TP8 × PP1。 |
| D 节点 | 80.48.29.134，容器 `wl_1011_new`，DP8 × TP2 × PP1。 |
| 模型 | `/mnt/share/weight/DeepSeek-V4-Flash-DSpark-w4a8-scope-ci`；沿用来源的 W4A8、DSpark/MTP 和模型专用参数。 |
| 通信 | A3 HCCS、`device_sdma`、`ASCEND_ENABLE_USE_FABRIC_MEM=1`；两节点的通信域标识及网卡设置需与现场一致。 |
| 池化 | memcache MetaService 位于 P；P、D 各一个 standalone LocalService 贡献存储；vLLM worker 的 `dram.size=0GB`，只接入池。 |
| DRAM/SSD | standalone 每节点 DRAM 5GB，SSD 使用 UBS IO 和本机 `/dev/loop0` 测试块设备；实际后备文件所在介质另行核实。 |
| 起点 | 已装好依赖并能使用原模型的对应容器。镜像和包版本身份需要按 02 补齐后固定，不能声称支持任意新镜像。 |

这套参数需要每节点 16 个可见 die。64G 是平台场景名称，不能把 P 的 TP8 简化理解为只占 8 张物理卡中的一半。设备分组和端口映射见 04。

## 2. 服务与数据路径

```text
客户端 → P 节点上的 PD 代理 :8080
             ├─ Prefill：P 的 DP0/DP1，API :8000/:8001
             └─ Decode：D 的 DP0..DP7，API :8101..:8108

PD KV 传输：MooncakeHybridConnector（P producer → D consumer）
外部 KV 池：AscendStoreConnector（backend=memcache）
                         ↓
             P 上的 MetaService :5000/:6000/:8004
                         ↓
         P、D 的 standalone：DRAM → UBS IO SSD 层
```

MooncakeHybridConnector 在此承担 PD 传输；外部池后端是 memcache。检查 PD 请求完成、外部命中以及 DRAM/SSD 分层命中时，分别观察对应的路径，避免把其中一个当成另一个的证据。

## 3. 覆盖范围

| 编号 | 需要检查的行为 | 判据 |
| --- | --- | --- |
| V1 | 服务拉起与真实推理 | Meta、两个 standalone、全部 P/D API、代理就绪；代理实际请求完整返回；目标连接器和 P→D 路径可核对。 |
| V2 | DRAM 保存与命中 | 在独立请求窗口内，能够核对 KV 写入/分配和后续 DRAM GET/命中；请求成功，无相关未处理 PUT/GET 错误。 |
| V3 | SSD 保存 | 能对应本轮工作集的 SSD 写入/落盘证据，或后续真实 SSD 命中所证明的先前落盘；容量注册、空盘初始化不作为保存证据。 |
| V4 | SSD 命中 | 同一运行期间的 `memcache_get_hits_ssd_total` 有有效正增量，相关请求完成，并排除其他任务、计数器重启和上层缓存的干扰。 |

默认先做短请求，再做少量前缀复用，必要时用来源已采用的 128K 输入、1 token 输出、多前缀工作集触发 SSD。无需固定 DRAM/SSD 命中占比，不要求性能收益对比、完整模型精度、跨重启保留或 profiling。

## 4. 测试设计

1. 保持本地 HBM prefix cache 关闭，P/D 共用同一模型、哈希种子和连接器拓扑。
2. 对固定前缀先预热，再发共享前缀的正式请求；P 是 DP2，预热覆盖和所有 P API 的指标均需核对。
3. 先用少量请求检查 DRAM。此时 SSD 未参与属于可能的正常现象，不能把单条重复请求直接作为 SSD 测试。
4. SSD 触发采用多前缀工作集和 DRAM 高/低水位 80/70。5GB/节点是来源选择的起点，是否发生淘汰取决于实际工作集，必须通过观察确认。
5. 每阶段对比前后指标。实际输入 token 数以服务/客户端结果为准，`input_len=131072` 是工具构造参数，chat 模板可能增加开销；`max-model-len=134144` 留有余量。
6. 未触发 SSD、缺少分层指标或请求失败时，按 05 判定，不用历史成绩填补本次检查。

## 5. 执行前须落实的事项

实际镜像/软件身份、代理实现、GSM8K 源数据、测试 Python 环境、设备健康、大页和磁盘归属都属于运行前条件。相应取得方法与缺口见 02、04。任何新运行只使用本次确认的空闲资源；旧日志里的运行状态不代表当前现场。