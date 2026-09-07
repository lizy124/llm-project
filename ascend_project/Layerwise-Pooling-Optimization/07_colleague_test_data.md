# 同事测试数据：DSV4-Flash Layerwise 池化（A3 / 16卡 / PD混部）

> 来源：同事 aisbench 测试表格全量数据（test_data.txt，742 行，2026-09-03 ~ 09-05 共 12 轮）
> 整理时间：2026-09-07
> 说明：分"单前缀/多前缀"两大段；所有轮次模型/硬件/负载框架一致，仅前缀数量与 KV 策略（prefix cache / 池化 / layerwise）不同；表格经 Excel 导出产生的双引号转义（`""`）已还原

---

## 1. 环境与公共配置

| 项 | 值 |
|---|---|
| 模型 | DS-V4-Flash（权重 `/mnt/lsy/deepseekv4-flash-w8a8-mtp`，w8a8 + MTP） |
| 产品形态 | A3 |
| 组网 | PD混部，16 卡（TP4 × DP4） |
| MAX_MODEL_LEN / MAX_NUM_BATCHED_TOKENS / MAX_NUM_SEQS | 1048576 / 10240 / 64 |
| 负载框架 | 设计命中率 90、总请求数 128、并发数 32、请求频率 0；输入/输出 131072/1（第3次单前缀为 16000/1） |
| block_size | 32 |
| MTP | `num_speculative_tokens: 3` |
| 调度 | `--async-scheduling`、`--enforce-eager`、cudagraph FULL_DECODE_ONLY |
| 端口 | 8100 |

### 1.1 四个配置变体（其余参数完全一致）

| 变体 | prefix cache | 池化 | layerwise | 关键差异 | vllm log 文件名 |
|---|---|---|---|---|---|
| **layerwise 池化** | 否 | 是 | 是 | `--no-enable-prefix-caching` + AscendStoreConnector(memcache, `use_layerwise: true`, `layerwise_prefetch_layers: 3`) | log_p.log / log_p_layerwise.log / log_p_test_layerwise.log |
| **非layerwise 池化** | 否 | 是 | 否 | 同上但 kv_connector_extra_config 无 use_layerwise | log_p_unlayerwise.log |
| **HBM 基线（单前缀）** | 是 | 否 | 否 | `--enable-prefix-caching`，无 kv-transfer-config | log_p_HBM.log |
| **HBM 基线（多前缀）** | 是 | 否 | 否 | 同上（重新拉起服务后跑） | log_p_HBM_morePrefix.log |

### 1.2 layerwise 池化完整启动脚本（其余变体按 §1.1 差异修改）

```bash
# ============ Configurable ============
TP_SIZE=4
DP_SIZE=4
MAX_MODEL_LEN=1048576
MAX_NUM_BATCHED_TOKENS=10240
MAX_NUM_SEQS=64   # 并发数
# export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm
# export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm-ascend

# MemCache config paths
export MMC_LOCAL_CONFIG_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/memcache_hybrid/config/mmc-local.conf

export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export TASK_QUEUE_ENABLE=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
# export HCCL_INTRA_ROCE_ENABLE=1

# ---------- 1. Environment Variables ----------
export PYTHONHASHSEED=0
export VLLM_USE_V1=1
# export ASCEND_BUFFER_POOL=4:8
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_BUFFSIZE=1024
export VLLM_ASCEND_APPLY_DSV4_PATCH=1
export HCCL_OP_EXPANSION_MODE="AIV"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# ---------- 5. Launch vLLM ----------
vllm serve /mnt/lsy/deepseekv4-flash-w8a8-mtp \
    --host 0.0.0.0 \
    --served-model-name dsv4 \
    --enable-expert-parallel \
    --no-disable-hybrid-kv-cache-manager \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --quantization ascend \
    --reasoning-parser deepseek_v4 \
    --model-loader-extra-config '{
     "enable_multithread_load": true,
     "num_threads": 128
     }' \
    --no-enable-prefix-caching \
    --tensor-parallel-size "${TP_SIZE}" \
    --data-parallel-size "${DP_SIZE}" \
    --enforce-eager \
    --model-loader-extra-config '{
     "enable_multithread_load": true,
     "num_threads": 128
     }' \
    --port 8100 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --block-size 32 \
    --gpu-memory-utilization 0.90 \
    --api-server-count 1 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"num_speculative_tokens": 3,  "method":"mtp","enforce_eager":true}' \
    --additional-config '
    {"ascend_compilation_config":{
        "enable_npugraph_ex":true,
        "enable_static_kernel":false
        },
    "enable_cpu_binding": true,
    "enable_dsa_cp": true,
    "enable_flashcomm1": true}' \
    --kv-transfer-config '{
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "lookup_rpc_port":"0",
                    "backend": "memcache",
                    "use_layerwise": true,
                    "layerwise_prefetch_layers":3
                }
            }' > log_p.log 2>&1
```

> 原脚本含两处重复的 `--model-loader-extra-config`（原样保留）；HBM 变体将 `--no-enable-prefix-caching` 换为 `--enable-prefix-caching` 并删去 `--kv-transfer-config` 整段。

## 2. 单前缀结果（前缀数量 1）

| 轮次 | 策略 | 输入长度 | 实际命中率 | MTP采纳率 | aisbench 时间 | 极光 task |
|---|---|---|---|---|---|---|
| 第1次 | layerwise 池化 | 131072 | 87.50% | — | （未记录） | — |
| 第2次 | layerwise 池化 | 131072 | 87.50% | — | 20260903_210818 | 96b65f08 |
| 第3次 | layerwise 池化 | 16000 | 76.78% | — | 20260905_114216 | 96b65f08 |
| 修复内存后 | layerwise 池化 | 131072 | 87.50% | — | 20260905_224238 | 96b65f08 |
| （无序号） | 非 layerwise 池化 | 131072 | （空，未记录） | （空） | 20260903_234013 | bccb94a9 |
| 第1次 | HBM prefix cache | 131072 | 85.45% | — | 20260904_090612 | 049ac517 |
| 第2次 | HBM prefix cache | 131072 | 85.45% | #NAME? | 20260904_100457 | 049ac517 |

## 3. 多前缀结果（前缀数量 32，输入 131072）

| 轮次 | 策略 | 实际命中率 | MTP采纳率 | aisbench 时间 | 极光 task |
|---|---|---|---|---|---|
| 第1次 | layerwise 池化 | 0.75% | — | （未记录） | — |
| 修复内存后 | layerwise 池化 | **87.50%** | — | 20260905_232330 | 96b65f08 |
| （无序号） | 非 layerwise 池化 | （全空，无任何数据） | | | |
| 第1次(重新拉起服务) | HBM prefix cache | **6.15%** | 没有值 | 20260904_103224 | 577509af |

## 4. 日志路径

- vllm log（容器内）：`/tmp/inference/<task-id>/mixed_single0_0/<log文件>`，log 文件见 §1.1 末列
- aisbench log：`/usr/local/python3.11.10/lib/python3.11/site-packages/outputs/default/<时间戳>`
- 极光平台 task：
  - `96b65f08-ccd6-4c7a-aca2-095c0479b617`（layerwise 池化全部轮次）
  - `bccb94a9-7a15-413e-967b-59c1ab11f83f`（非 layerwise 池化单前缀）
  - `049ac517-ea21-419d-8b2c-4786f1f865e1`（HBM 单前缀两轮）
  - `577509af-4048-48ea-afae-d7b0a4781b94`（HBM 多前缀）
  - 链接格式：`https://jiguang.ascend.huawei.com/inference/task/<task-id>`
- 所有轮次 aisbench 结果截图列均为 `#NAME?`（Excel 引用失效），**全表无任何 TPS/TTFT 数据**

## 5. 关键发现

1. **多前缀场景 HBM prefix cache 命中率崩塌至 6.15%，layerwise 池化维持 87.50%**（32×131072 ≈ 4.2M token KV 远超 HBM 容量）。这是"容量换性能"价值的最直接量化：多前缀形态下池化不是劣化项，而是从 6.15% 命中提到 87.50% 的收益项。与我们 165 侧结论（w4a8 TP8 全机 KV 仅 274.5K token ≈ 2.1 条 128K 前缀）互为印证。
2. **单前缀：layerwise 池化 87.50% vs HBM 85.45%**——命中率层面池化不输 HBM 基线（差距 2.05pct，非容量因素，推测与两侧对齐/粒度差异有关）。
3. **16K 单前缀 76.78% ≈ 12288/16000 = floor(16000/4096)×4096/16000**，特征与 `cache_transfer_granularity=4096` 尾块截断完全吻合。对照本组 165 实测（HBM prefix cache 路径前缀 <16384 恒 0% 命中）：**池化路径无 16384 硬边界，粒度更细（4096），短前缀场景仍有大部分收益**——正是 layerwise 池化相对原生 prefix cache 的发力点。
4. **"修复内存"的效果集中在多前缀**：多前缀 layerwise 0.75% → 87.50%；单前缀修复前后均 87.50%（09-03 第1/2次即达标）。（09-07 闭环：即 PR #15854，见第 7 条）
5. 128K 输入下 87.50% 与设计 90% 差 2.5pct，单双前缀、修复前后均稳定复现——是系统性截断（粒度/对齐），非偶发。（09-07 闭环：90%×131072=117,965 token 共享前缀，按 4096 粒度向下取整=114,688=131072×7/8 → **87.50% 是粒度截断的必然值**）
6. MTP 采纳率数据全表缺失（空/#NAME?/没有值），无法评估 MTP 叠加效果。
7. **栈来源已确认（2026-09-07 同事答复）：upstream main 2026-08-27 最后 commit 基线 + PR #15442 + PR #15854**，两个 PR 均未进 upstream main（pool_165 实测 main 至今仍带 1-main 断言；#15854 状态 Closed）：
   - **PR #15442**（作者 lsy0214，即同事本人，/mnt/lsy/ 权重主人）：`build_layerwise_reuse_layout` 多 main spec 适配——`.attn` 选为 main，其余非 indexer spec 进 `extra_main_specs`，indexer 可选，单 spec 层保留 fast path，全 indexer 层仍 raise；带完整 UT。**这正是 [06_handover.md](06_handover.md) §1.3(c) 评估的"中大型改造"——已有人做完且公开**
   - **PR #15854**（tyy0829）= 表格"修复内存"：layerwise 原先把 SWA/compressor 组**全部**块入库（单条 128K 请求 ~45.6 GB 池内存，98% 浪费在必要子集仅 ~1.4 GB 的 state/SWA 组），修复为三侧（save/hit check/load）对齐非 layerwise 的 reachable-store 语义 → 45.6→1.4 GB。这解释了多前缀 0.75%→87.50%（32 前缀 × 45.6 GB 撑爆池，修复后放得下）
   - PR #15854 附带 **MTP trim 修复**：16K prompt 量化命中 12288 恰为末块 [12288,16384) 起点，旧阈值误判"进入末块"trim 到 8192（51.2%），修复后满载 12288 → 76.78%。**第3次 16K 数据就是带此修复测的，16K 之谜闭环**
8. **PR #15442 内嵌初步性能数据**（自测，8x910B3、memcache host_shm）：layerwise **OFF 146.70 / ON 53.76 / ON+prefetch 55.37 tok/s**——ON 劣化 63.4%，prefetch 仅 +3%。注意这是"使 layerwise 刚能跑通"的 PR 自测数（早于 #15854 修复、host_shm 非完整 DRAM 池形态），只能当方向性参考；正式劣化% 仍缺，正是待补的核心数据

## 6. 备注与结论

1. **DSV4-Flash × memcache × layerwise 在该环境可正常启动运行**。本组 165 环境的 5-spec 启动崩溃（[06_handover.md](06_handover.md) §1.3）在该环境未出现。
2. **根因已定位（2026-09-07）：同事栈 = upstream main 2026-08-27 基线 + PR #15442（多 main spec 布局适配）+ PR #15854（reachable-store/MTP trim 修复），两个 PR 均未进 upstream main（pool_165 实测 main 仍带 1-main 断言），故 165 裸栈必崩、配置对齐无解**。逐项核实：
   - 165 栈 = upstream main 最新（33e849499，fetch 无更新），1-main 断言在 main 及所有 release 分支都在，树上无任何 DSV4+AscendStore E2E 测试 → 165 配置对齐无解
   - **`VLLM_ASCEND_APPLY_DSV4_PATCH=1` 是历史 flag**：2026-06 vllm-ascend #10333 已移除（DSV4 hybrid coordinator 行为已自动化，spec 带 `model_version="deepseek_v4"` 即生效），设它是旧习惯 no-op，与 5-spec 崩溃无关
   - `enable_dsa_cp`/`enable_flashcomm1`/`--no-disable-hybrid-kv-cache-manager` 在 165 栈均支持，但 `deepseek_v4.py` spec 创建无条件分支，每层 5 spec 结构改不掉
   - 两环境硬件同规格（A3×16），**核心差异是栈**（同事 python3.11.10 site-packages + 极光平台托管 vs 165 python3.12.13 editable /vllm-workspace）
3. **165 上有同款权重 `DeepSeek-V4-Flash-w8a8-mtp`**（/mnt/weight/），后续复测同形态可用
4. 该配置在 KV connector 场景下使用了 `PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"`（本组已知教训：expandable_segments 与 KV connector MR 注册存在不兼容风险），而"修复内存"内容未记录——两者是否相关待确认
5. 并发表格值 32，与脚本 `MAX_NUM_SEQS=64` 不一致，以表格负载口径为准（32）
6. **栈来源已确认（09-07），剩余待索取**：① **各轮 TPS/TTFT 数据**（aisbench 截图全为 #NAME?，全表零性能数据——劣化% 计算的唯一缺口）；② MTP 采纳率实测值。（"修复内存"内容已由 PR #15854 闭环，栈来源已由两 PR 闭环，无需再问）
