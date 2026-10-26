
# 05 简单验证过程与判定方法

本章保留五主题文件名，内容只包括执行顺序、必要观察项和判定方法；不记录历史成绩、性能表、命中率成绩或逐轮测试结果。服务启动操作见 [04](04_verification_env_and_launch.md)，测试调用与池化指标脚本见 [03](03_verification_scripts.md)。AISBench 内部实现独立整理。

## 1. 验证次序

| 阶段 | 做什么 | 观察什么 | 可以确认什么 |
| --- | --- | --- | --- |
| T0 就绪 | 检查 Meta、两端 standalone、全部 P/D API、代理和连接器 | 当前进程、本次日志、health、模型名及池段 | 服务与目标功能已配置和注册；尚不能确认 KV 保存或命中。 |
| T1 短请求 | 经代理发送一次实际请求 | 有效响应、完整生成、P/D 路径及相关错误 | 基本推理可用；不能仅凭 health 或 HTTP 200 判断 PD 传输成功。 |
| T2 DRAM | 固定前缀先预热，再发少量共享前缀请求 | DRAM 写入/分配、DRAM hit 前后差、所有 P API 的 external query/hit | 目标工作集写入外部 DRAM 并被读取复用。 |
| T3 SSD | 在同一服务生命周期内扩大多前缀工作集，触发 DRAM 淘汰并复查 | SSD 写入/落盘、SSD 分层命中正增量、请求完成及错误 | KV 实际进入 SSD 层；有有效读取证据时进一步确认 SSD 命中。 |
| T4 退出 | 按 04 逆序停止本轮拥有的进程并恢复准备项 | 进程、端口、NPU、loop 与大页归属 | 本轮占用已释放，原验证材料和其他任务保留。 |

## 2. T0/T1：服务与 PD 请求

T0 要检查 P 的两个 API、D 的八个 API，而不是任选一个端口代替整个 DP 组。代理列表与这些端口对应，served model 为 `ds`；日志应确认 MultiConnector 中同时启用了 MooncakeHybridConnector 与 AscendStoreConnector(memcache)。

短请求从可访问代理的 Linux 终端执行：

```bash
curl --noproxy '*' --max-time 300 -fsS \
  http://80.48.29.133:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ds","messages":[{"role":"user","content":"Reply with hello only."}],"temperature":0,"max_tokens":16,"stream":false}'
```

检查 JSON 内是否为有效完成响应，而不是错误对象或不完整返回。通过可关联的 P/D 请求日志、KV 传输参数和加载行为核对实际 P→D 路径。若出现回退重算、PUT/GET 失败或传输超时，即使最终有文本也不能直接确认 PD 数据面通过。

## 3. T2：DRAM 保存与命中

1. 选取本轮专属前缀/seed，在安静的测试窗口运行，避免其他客户端同时贡献指标。
2. 保持本地 prefix cache 关闭。测试前后分别读取 Meta 的原始 `/metrics` 和两个 P API 的 `/metrics`；指标不存在应报缺项，不应当作零。
3. 用前缀工具做少量 warmup 与正式请求：在全新工具工作副本中设置 `CONCURRENCY=1`、`SEED=901`、`PREFIX_NUM=1`、`DATA_NUM=4` 后，执行 03 的同一条 AISBench 命令。输入仍为 128K、输出为 1；前缀数量 1、P DP2，核对生成的 warmup 数量及预热覆盖。
4. 等待异步保存完成，观察相应写入/分配信息；随后的共享前缀请求应完整成功，并使 DRAM 分层命中与 external hit 出现有效增量。
5. 若工作集已溢出 SSD，本轮成为混合层测试，不能把总 external hit 全部标为 DRAM 命中。依分层计数分别判读。

优先使用已核实的 PUT 成功、相应 key 的 DRAM 状态或分配记录来检查保存；在受控工作集下，后续真实 DRAM 命中也证明此前存在可读取的 DRAM 数据。仅创建池容量不证明保存。

P API 指标获取示例：

```bash
curl --noproxy '*' -fsS http://80.48.29.133:8000/metrics
curl --noproxy '*' -fsS http://80.48.29.133:8001/metrics
```

来源的 AISBench `config.py` 只列 P:8000 作为 POD_INFO，不能据其表格自动宣称覆盖了两个 P API。手工检查所有相关 engine 的增量，或在使用前显式调整本轮独立工具副本的采集范围。

## 4. T3：SSD 保存与命中

保持 Meta 和 standalone 不重启、不重新初始化盘。配置为每机 DRAM 5GB、SSD 开启、水位 80/70；通过多个不同前缀扩大工作集，再访问已写入的相同前缀。来源采用的触发负载为：

```text
input_len=131072，output_len=1，repeat_rate=0.9，dp=2
concurrency=32，seed=201，prefix_num=32，data_num=64
```

04 给出该组完整调用。先通过短请求与 T2，再运行该工作集；无需复跑全部性能对照组。单前缀少量请求可能一直留在 DRAM，SSD hit=0 时先判断是否触发目标场景。

需要观察：

- SSD 层容量和磁盘状态可用，但只把它们当作注册/健康检查。
- 是否出现对应工作集的 SSD 写入、淘汰下沉或后续分层 SSD 命中。
- 所有相关请求是否完成，是否有 PUT、GET、UBS IO 或 PD 传输错误。
- 窗口内是否有其他客户端、服务重启、key 空间复用等干扰。

只要求“SSD 确实保存”时，拿到直接落盘证据即可，不强制命中；本套流程也提供命中检查，以便需要时确认读取路径。若分层 SSD hit 有有效正增量，且工作集归属与请求成功均可核对，它也证明这些数据此前已落入 SSD；这不等同于跨重启持久化或指定命中占比。

## 5. 指标读取和差值

先从实际版本的 Meta `/metrics` 确认存在：

```text
memcache_get_hits_dram_total
memcache_get_hits_ssd_total
memcache_ubs_io_disk_capacity_bytes
memcache_ubs_io_disk_status
```

这些名称来自选定来源，具体计数单位、磁盘状态码和标签仍需结合被测版本核对。指标改名或缺失时，先定位对应指标/日志，不能按名称推断为功能失败。

```text
dram_delta = dram_after - dram_before
ssd_delta  = ssd_after  - ssd_before
```

需要比例时，分母为同一窗口 `dram_delta + ssd_delta`；这是池后端分层读取占比，不等于输入重复率或 vLLM 的前缀 token 命中率。本流程不设比例目标。

03 的 `tier_delta.py` 接受前后两个原始 Prometheus 文件，逐序列检查后打印分层增量；缺指标、序列变化、非有限值或计数器下降时拒绝给出有效增量。仍须人工确认整个窗口服务没有重启，因为重启后计数器可能已增长到旧值以上。

## 6. 判读边界

| 观察 | 应如何处理 |
| --- | --- |
| 容量非零、初始化成功 | 只能确认层已注册，继续检查保存与命中。 |
| DRAM/SSD 对应增量为正，相关请求成功且无干扰 | 可确认相应分层读取；结合写入证据判断保存。 |
| SSD 增量为零，缺少淘汰/落盘证据 | 标为未触发或证据不足，调整工作集后检查。 |
| 请求失败、实际零发送、日志为旧轮次、统计缺失 | 本轮无效或失败，不用脚本退出码覆盖这些事实。 |
| 只有总 external hit、磁盘 I/O 或延迟变化 | 不足以单独确认 SSD 保存与命中。 |
| 表内 TPOT=99999 等占位 | 1 token 输出不用于 TPOT 判定；按字段含义检查，不把所有占位误判为请求失败。 |

本套文档保留上述方法即可。执行时用临时日志、请求明细和前后指标完成检查，不要求把本次测试成绩、历史吞吐或运行汇总写回这五份文档。