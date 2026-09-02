# map_165 E2E 计划 — PR #15367 服务器验证

> 规则层:`../../llm-project/transfer_data/envs/create_env.md`(环境)+ `verify_guide.md`(验证)
> 需求层:`PR-15367 record.md` §服务器验证清单 + `server-validation.md`
> 本文档 = 165 实例执行计划;结论产出归档到 `record_final/`。

## 0. 验证对象

- 分支:`lizy124/vllm-ascend:refactor_layerwise_part1` @ `2a239d18a`(7 commits)
- 重构内容:GVA 逻辑协议对象化、注册表数据绑定、双参 gate;核心回归点 = 多 connector 场景 `kv_producer`/`kv_consumer` 角色无属性崩溃(#14465 型)

## 1. 验证矩阵(修正后,verify_guide §8.1 v2 裁决)

| # | 场景 | 模型 | backend / 配置 | 核心判定 | 优先级 |
|---|---|---|---|---|---|
| 1 | MultiConnector PD 分离 | DeepSeek-V2-Lite | memcache + layerwise,P TP=4 / D TP=4 + proxy | 请求成功率 100%;**无 AttributeError** | 高 |
| 2 | memcache layerwise 冒烟 | DeepSeek-V2-Lite | memcache + `use_layerwise=true` | debug `load_gvas: valid_gvas>0`;metrics `hit_tokens>0` + 三维证据链 | 高 |
| 3 | mooncake 非 layerwise 零回归 | Qwen3-32B | mooncake(50088) | 三维证据链 + 行为与 main 基线一致 | 中 |

模型裁决:
- 场景 1/2 验证 layerwise **传输** → 必须 MLA/SFA 模型(`attention_v1` 零集成 layerwise wait/save,verify_guide §8.1);Qwen3-32B 若用只能写"布局通过"
- 场景 3 与 layerwise 无关 → Qwen3-32B(/mnt/weight/Qwen3-32B,历史冒烟同款)
- DSV2-Lite 在官方支持矩阵;DSV4 禁用(NSA 撞硬限,PR14465 已踩)

## 2. 165 环境事实(已盘点,详见 PREP.md)

- 镜像:`7f06feda13d3`(6 容器主流,cxy_cann9.1.0 同款,tag 待补)
- NPU:12 卡全空闲;磁盘 2.2T
- 模型:`/mnt/weight/Qwen3-32B`(62G)、`/mnt/weight/DeepSeek-V2-Lite(-Chat)`
- 数据:`/mnt/share/c00814587/.../GSM8K.jsonl`(1319)/ `gsm8k-lite.jsonl`(32)
- 网络:直连;无网时本地 scp

## 3. 环境准备步骤

### 3.1 建容器 `refactor_165`
1. `docker inspect cxy_cann9.1.0` 抄本机挂载/设备/网络惯例
2. 按 create_env.md §3 模板创建;挂载必含 `-v /mnt:/mnt -v /home/lizhongyang/map_165:/workspace`
3. 预检:NPU 占用(npu-smi)、端口(§3.3 表)

### 3.2 代码与版本配对
1. clone:`/workspace/vllm-ascend` @ `2a239d18a`;`/workspace/vllm` @ 配对 commit
   (查 `vllm-ascend/.github/vllm-release-tag.commit` / `vllm-main-verified.commit`)
2. 安装(verify_guide 165 先例):容器内 `uv pip install --python /venv/.../python /vllm-workspace/vllm` 再 `/vllm-ascend`
   用 uv.lock 对齐 torch/torch_npu/torchvision,**禁手动装 torch**;
   A3 若报 `libbsd.so.0`:`apt-get install -y libbsd0`(需配源)
3. 版本断言:`pip list | grep -E 'vllm|torch'` 入档

### 3.3 池化前置(verify_guide §3.7 探针项)
| 项 | 操作 |
|---|---|
| hugepages | 容器内 `sysctl -w vm.nr_hugepages=200000`(memcache device transfer 硬性前置,缺了 abort) |
| 后端包 | 查 `pip list | grep -iE 'memcache|memfabric'`;缺则按 env_install/7_kv_backends_install.md 装 |
| mmc-local.conf | 拷模板改 `protocol=device_sdma`(A3 HCCS);`dram.size` 按 62GB HBM 对齐;**禁 root 直接改系统 conf** |
| MetaService | 独立拉起 + `ss -lnp` 探活 5000/6000/8000;`:8000/metrics` 实测可用 |
| 端口 | 50051 预检(DTMA 占用史)→ mooncake 用 50088;8004 服务口 |
| A3 环境变量 | `ACL_OP_INIT_MODE=1`、`ASCEND_ENABLE_USE_FABRIC_MEM=1`(MIX 场景) |

### 3.4 指纹入档
改写 `pool_env_check.sh` 参数化(CPU_CORES/MEM_GIB/DOCKER_IMAGE/NPU chips 0-11,
`ENABLE_HOST_DOCKER=0`)→ 输出 `record_final/env_fingerprint_165.md` + 脚本入 `archive/`。

## 4. 通用启动铁律(每次拉起必守)

- `--no-enable-prefix-caching`(PR15072 起池化 V1 接管;双开 GPA 先截)
- `PYTHONHASHSEED=0`(token hash 不稳定会吞 prefix hit)
- `VLLM_ENGINE_READY_TIMEOUT_S=2400`;`--load-format dummy` 仅调试用
- 服务 READY 后**先查** `master_active_clients{role="pool_client"}==DP×TP`,再发请求
- warm-up 预热(首请求不止统计,还建池连接),用真实权重,禁 dummy 预热
- prompt ≥ 2×granularity 块(layerwise 2 倍块),禁默认 max_tokens=16 截断
- NPU 物理卡选择(`ASCEND_RT_VISIBLE_DEVICES`),每轮 vLLM PID 唯一

## 5. 场景执行要点

### 场景 1:MultiConnector PD 分离(最高优先级)
- **注意:verify_guide 未覆盖 PD 路径**;按官方 kv_pool 文档 §2.3/§3.5 补:
  P 实例 `kv_producer` 角色 + D 实例 `kv_consumer` 角色,MultiConnector 包装,
  layerwise 专用 proxy(`/v1/metaserver`,`--host` 禁 0.0.0.0)
- 拓扑:P TP=4(chips 0-3)+ D TP=4(chips 4-7),GSM8K prefix-cache 请求
- 回归点:重构前 `kv_producer`/`kv_consumer` 角色访问不存在属性崩溃;mock 路径已删,
  任何 AttributeError 即真失败
- 判定:请求成功率 100%;两侧日志无 AttributeError/Traceback;KV 跨实例命中
  (D 侧 external hit > 0)

### 场景 2:memcache layerwise 冒烟
- 单实例 TP=4(chips 0-3),kv-config `backend=memcache` + `use_layerwise=true`
- granularity 用更大块;长前缀请求(≥2×layerwise 块)发 2 轮
- 判定:debug 日志 `load_gvas: ... valid_gvas=N>0`;`external_prefix_cache_hit_tokens>0`;
  三维证据链:① `master_put_success_tokens>0` ② `master_get_success_tokens>0`
  ③ `external_prefix_cache_queries ≈ 2× hits`(模型无关断言,verify_guide §8.5)
- 误区:`master_put_start_requests_total=0` 正常(batch 后端);hit rate 滑动窗口只判 >0

### 场景 3:mooncake 非 layerwise 零回归
- 单实例 TP=4,mooncake.json(`local_hostname` 从 `docker0` 改 `eth0`,port 50088)
- 判定:三维证据链通过;`master_get_success_keys/tokens > 0`;
  与 main 基线行为一致(对照项:启动参数、日志无新增告警)

## 6. 执行顺序

1. 容器 + 安装 + 探针指纹(§3)
2. 场景 2(单实例最简,先打通 memcache 链路)
3. 场景 3(mooncake 对照)
4. 场景 1(PD 最复杂,最后攻坚;缺脚本时按官方文档现写,入 `test/`)

## 7. 产出归档

每场景独立目录 `record_final/<场景>_<日期>/`:启动命令全文、服务日志关键段、
metrics 快照、判定结论(三维证据链逐行对账)、版本表。
结论回写 `PR-15367 record.md` §4 + PR 描述。

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| PD 路径无现成脚本 | 按官方 kv_pool §2.3/§3.5 现写;失败先按 record.md 排查清单定位 |
| memcache 包版本错配 | 安装后先 `import` 冒烟 + MetaService 探活再启 vLLM |
| NPU/端口与他人冲突 | 每轮启动前重查 npu-smi + ss;端口冲突换备口不改他人容器 |
| DSV2-Lite 无 Chat 模板差异 | 用 `-Chat` 变体或 base 模型 completions 接口 |
