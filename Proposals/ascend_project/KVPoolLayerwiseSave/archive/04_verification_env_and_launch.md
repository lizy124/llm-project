# 04 — 环境、服务启动与复现边界归档

归档日期：2026-10-24。本次没有创建、连接或停止服务器环境。下列环境按身份分开；历史 129/130 平台运行与 133/135 复现脚本不是同一实验。

## 1. 环境身份

| 项目 | E1：主要历史现场 | E2：留存复现脚本 |
|---|---|---|
| P / D | 80.48.29.129 / 80.48.29.130 | 80.5.9.133 / 80.5.9.135 |
| 容器 | `yxf_vllm_b113`，日志标签 docker-b113 | 脚本存在，实际容器身份及成功运行未独立留证 |
| P 并行 | DP4×PCP4×TP1 | DP4×TP4，不能当成 PCP4 等价对照 |
| D 并行 | DP16 | DP16×TP1 |
| P/D API | 8000～8003 / 8000～8015 | 8910 / 8920 |
| 代理 | 129:1999 | 133:8100 |
| memcache Meta | 129:5000，config store 6000 | 135:5000、6000，metrics 8004 |
| 池 | device_sdma、DRAM；无 SSD 数据面 | 每节点 standalone 20GB，engine proxy 配置 0GB |
| 模型 | DeepSeek-V4-Flash-0731-w8a8，max length 1048576 | 同名权重目录；精确权重哈希未留存 |
| 源码 | `/vllm-workspace/vllm-ascend`，v0.30.0 系列，具体版本见 02 | 启动前需另行固定配对版本，归档未证明已运行哪个 commit |

10-24 synchronize 是 E3：129/130 的 b118 标签，属于独立故障轮次；不要与 b113 上的根因修复采样合并。`task_v4_pro_logs` 是另一组 137～140 节点材料，本归档仅登记原件，未重新完成根因分析。

E1 镜像名称记录含 daily-releases-v0.30.0-a3-openeuler-cann9.2.0，但完整 digest、vLLM 精确 SHA、memcache 包精确版本不足。只能声明留存材料支持的复现范围。

## 2. 从明确起点准备

E1 的起点是历史已有容器与平台生成的启动目录。完整 `launch_online_dp.py`、`run_dp_template.sh` 未取得，不能凭命令摘要声明任意新环境均可复建。恢复实验前应固定源码、配对版本、镜像、权重和 AISBench 工具版本，再建立新轮次。

E2 可使用 03 第一部分的完整配置与规范化启动脚本，但它们是整理版，仅做本地文本核对，未重新在 NPU 上验证。准备工作包括：

1. 在明确授权的容器中安装配对的 vLLM、vLLM-Ascend、CANN 和 memcache_hybrid；核对模型路径。
2. 准备工作根，例如 `/tmp/kvpool_repro_133_135`，按本任务 03 的 config、launch、tests 结构复制文本资产；不复制外置日志。
3. 安装现有 AISBench 与 `aisbench_test.py` 工具。工具完整源码/精确版本未随任务取得，是复现缺口；配置 `aisbench_config.py` 需作为该工具工作目录的 `config.py`。
4. 设置 `KVPOOL_CONFIG_DIR=/tmp/kvpool_repro_133_135/config/repro_133_135`、`KVPOOL_LAUNCH_DIR=/tmp/kvpool_repro_133_135/launch/repro_133_135`。二者由调用者提供，启动脚本不会创建目录。
5. 核对网卡 `enp194s0f0`、IP、权重、库及代理脚本路径是否属于当前环境。改变这些条件后应作为新运行记录。

## 3. 资源和端口检查

在宿主机确认所需卡与内存空闲、服务归属和目标端口，遵循共享服务器纪律。E2 需 135 上的 5000/6000/8004、133 的 8910/8100、135 的 8920、各节点 12321 及 connector 端口；E1 使用其原记录端口，不能混套。

不得仅因旧文档记录进程名就执行全局 pkill。停止或恢复必须使用本轮明确记录的容器、服务及 PID；本次归档没有进行此类操作。

## 4. 服务启动顺序

### 4.1 E1 平台历史入口

原记录在容器生成目录运行：

```text
P: python launch_online_dp.py --pcp-size 4 --dcp-size 1 --dp-size 4 --tp-size 1 --pp-size 1 --dp-size-local 4 --dp-rank-start 0 --dp-address 80.48.29.129 --dp-rpc-port 12321 --vllm-start-port 8000
D: python launch_online_dp.py --dp-size 16 --tp-size 1 --pp-size 1 --dp-size-local 16 --dp-rank-start 0 --dp-address 80.48.29.130 --dp-rpc-port 12321 --vllm-start-port 8000
```

这些是留存调用记录，依赖缺失的平台脚本。关键条件为 P 的 `use_layerwise=true`、prefetch layers=3、max batched tokens=8192、block size=32、max seqs=16、chunked prefill，以及 MultiConnector 的 MooncakeHybrid+AscendStore；六组实际块大小为 `[128,4096,32,32,2,8]`。E1 的参数节录全文位于 03 的记录附录，不作为可直接启动的 shell 脚本。

### 4.2 E2 完整文本入口

先 Meta，再两节点 standalone store，再 P、D，最后代理。每个服务一个脚本，全文在 [03 第一部分](03_verification_scripts.md#part-config-launch)。以下为准备完成后的调用示例，没有在本次执行：

```bash
# 135，前台；日志重定向与后台管理由调用者处理
bash "$KVPOOL_LAUNCH_DIR/meta.sh"
# 133、135 各运行一个 store
bash "$KVPOOL_LAUNCH_DIR/standalone.sh"
# 133
bash "$KVPOOL_LAUNCH_DIR/p.sh"
# 135
bash "$KVPOOL_LAUNCH_DIR/d.sh"
# 133，在 P/D 都就绪后
bash "$KVPOOL_LAUNCH_DIR/proxy.sh"
```

整理版取消了启动脚本内的 cd 和 tee，配置/辅助模块路径改为显式环境变量；参数保留原文。外部代理文件 `/home/w00985460/script/vllm/load_balance_proxy_server_example.py` 未随任务取得，需按当前环境提供。

## 5. 就绪、冒烟与场景有效性

服务可访问、请求成功、目标路径触发是三个不同判据：

- READY：对应全部 API `/health` 返回 200，模型列表与引擎身份正确；E1 的 wait P/D 辅助脚本只适用于其地址。
- 冒烟：经本轮代理发请求成功，检查错误体；不能只以 curl 进程退出码判断推理成功。
- 正常池化：出现实际非零 hit/need-to-load、正常保存及可复用命中。比较同一 prompt 的未命中与命中输出，并记录服务日志证据。
- 复现/根因修复：长上下文、chunked prefill 和抢占恢复实际发生；关联同一请求的 new、marker、下一步 cached 分支与表，确认完整表没有被增量覆盖。

设备 event 等待、worker fence 和 snapshot 模式需要单独的目标判据，不能用零 GVA fatal 代替。

## 6. 测试入口与版本

E1 的 10-23 循环压测实际采用 03 第二部分的 `soak_aisbench_prefix_loop.sh`，`SOAK_HOURS=20`、`CASE_TIMEOUT=3600`，工具工作目录 `/home/aisbench_prefix`。实际记录约 11h32m。10-24 定向脚本为 `soak_mrv2_109k_10h.sh`，保留当时极端负载参数，不能把当前文件自动当作每个历史时刻都执行过的版本。

这两个历史循环脚本会在其专用工具目录重建数据、清理输出；本次未执行。再次使用前核对目录归属、日志位置及工具，不能在未知共享目录直接调用。

E2 的 run_case 脚本按自身目录运行 `aisbench_test.py`，因此需将选定脚本复制到已准备好的 AISBench 工具工作目录再调用。其参数与 E1 原压测略有差别，不能当作严格复现成功证据。c6 的“开测 1 分钟”注释是历史推论，不是本次实测判据。

本地统计复算使用 `analyze_soak_summary.ps1`，不依赖服务器或 Python；完整调用与输出见 [03](03_verification_scripts.md)、[05](05_verification_process_and_results.md)。

## 7. 停止、恢复与遗留状态

本次没有新增服务器进程，因此无需现场恢复。历史止血 v1、错误尝试及 v3 的部署和回滚仅作为记录留存；目前服务器状态未查询。

后续实验应记录本轮 PID/服务管理身份，再按代理、引擎、池服务的依赖关系停止；恢复代码使用固定基线和明确备份，检查新进程加载的路径、哈希与启动时间。不要沿用历史全局 kill/reset 命令。

当前复现限制包括精确依赖版本、E1 平台脚本、E2 外部工具/代理、纯 PR head、完整数据集和部分原始引擎日志。归档没有补造这些材料。
