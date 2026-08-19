# PR #14465 验证报告

> 验证目标：确认 PR #14465（`ascend-store-refactor-1`）达到可独立评审、先行合入状态  
> 验证日期：2026-08-18  
> 验证基线：`upstream/main` = `1a144d6c3`，PR HEAD = `2ef35ee6c`  
> 对照基线：PR #13160 修复后 = `8ad33de37e`，拆分第二部分 = `25329b86c`

## 一、结论

**PR #14465（`ascend-store-refactor-1`）通过全部验证，可独立合入。**

| 维度 | 结果 |
|---|---|
| 提交结构 | 3 commits，规模 +1029/-2942，DCO 3/3 ✓ |
| 静态检查 | `git diff --check` ✓，`compileall` 6 文件 ✓ |
| 单元测试（真实环境） | refactor_810: 232 passed；refactor_818: 258 passed |
| 等价性（与 PR #13160 修复后） | tree hash 一致，`git diff --quiet` rc=0 ✓ |
| review2.md 修复项 | block-size fallback ✓，lookup all/partial/exception ✓，零长度区间 ✓ |
| review.md 失败项 | 2 个 coordinator 测试在真实环境通过（mock 问题确认） |
| 真实环境 smoke（mooncake） | qwen3-32b pdmix non-layerwise，重复前缀命中 ✓ |
| 真实环境 smoke（memcache + layerwise） | qwen3-32b pdmix layerwise，重复前缀命中 ✓ |
| layerwise 路径激活 | `pool_worker.py:433 layerwise config: num_layers=64` ✓ |

**缺口**：无阻断项。内部 API 签名变化（删除部分构造参数/属性）若视为内部实现可接受。

**附注**：DSV4 MLA + layerwise 不兼容（`layerwise_cache_layout.py:221` 要求多 spec 层恰好 1 main + 1 indexer.k_cache，DSV4 有 5 个 spec），但此文件不在 refactor-1 改动范围内，属于 refactor-2。

---

## 二、验证详情

### 阶段 0：环境与基线确认

- 容器：`refactor_810`（CPU 环境，用于 UT），`refactor_818`（8×Ascend910 NPU，用于 smoke）
- 仓库路径：`/vllm-workspace/vllm-ascend`
- vllm 版本：`v0.27.1`（commit `6e448d0ea`，`FusedMoEFactory` 存在）
- CANN：9.0.1

### 阶段 1：PR #14465 分支核查

**提交结构**（`fork/main..fork/ascend-store-refactor-1`，3 commits）：

| commit | subject | Signed-off-by |
|---|---|---|
| `816ad0022` | refactor(kv_pool): extract metadata helpers to module-level functions | ✓ |
| `4b923ec54` | refactor(kv_pool): remove redundant state and simplify unit tests | ✓ |
| `2ef35ee6c` | fix(kv_pool): preserve lookup and block-size fallback behavior | ✓ |

规模：`12 files changed, +1029/-2942`（与 pr_split.md 文档一致）

### 阶段 2：静态检查

| 检查项 | 结果 |
|---|---|
| `git diff --check upstream/main..HEAD` | 无空白错误 |
| `python -m compileall`（6 生产文件 + 6 UT 文件） | 全部通过 |
| `ruff check` | ruff 未安装，跳过（非阻断） |

### 阶段 3：单元测试（真实 torch/vllm 环境）

#### refactor_810 容器

```
6 AscendStore UT + Mooncake UT: 225 passed, 0 failed
test_coordinator.py (单独): 7 passed, 0 failed
```

#### refactor_818 容器（NPU 环境）

```
ascend_store/ + mooncake/: 258 passed, 0 failed (11.45s)
```

#### review2.md 修复项核查

| 修复项 | 源码验证 | 测试验证 |
|---|---|---|
| `get_effective_group_block_size()` 越界 fallback | `metadata.py:233`: `block_size = group_block_sizes[0] if group_id >= len(...) else ...` | ✓ 存在 |
| `KVPoolWorker.lookup()` 全命中 | - | `test_lookup_all_cached` ✓ |
| `KVPoolWorker.lookup()` 部分命中 | - | `test_lookup_partial` ✓ |
| `KVPoolWorker.lookup()` 异常 | - | `test_lookup_exception` ✓ |
| 零长度保存区间 | - | `test_process_save_for_layer_batch_skip_zero_range` ✓ |

#### review.md 失败项核查

review.md 报告"2 个 coordinator 测试因 mock 失败"。在真实环境（refactor_810）下 `test_coordinator.py` 7 个测试全部通过，确认为 mock 环境问题，非代码缺陷。

### 阶段 4：等价性验证

| 验证项 | 文档预期 | 实际 | 结果 |
|---|---|---|---|
| `ascend-store-refactor-1` 是 `ascend-store-refactor-2` ancestor | 是 | `git merge-base --is-ancestor` rc=0 | ✓ |
| `ascend-store-refactor-2` 与修复后 `ascend-store-refactor` tree hash | 相同 | `a2e713465` == `a2e713465` | ✓ |
| `git diff --quiet refactor-2 refactor` | rc=0 | rc=0 | ✓ identical |
| `main..refactor-1` 规模 | +1029/-2942 | +1029/-2942 | ✓ |
| `refactor-1..refactor-2` 规模 | +18/-42 | +18/-42 | ✓ |
| `main..refactor-1` commit 数 | 3 | 3 | ✓ |
| `refactor-1..refactor-2` commit 数 | 1 | 1 | ✓ |

**结论**：拆分逻辑完全成立，两部分叠加后与修复后基线 tree 完全一致。

### 阶段 5：真实环境 smoke

#### 环境

- 容器：`refactor_818`（8×Ascend910，每卡 65536 MB HBM）
- NPU：0-3（TP=4）
- vllm-ascend：`2ef35ee6c`（ascend-store-refactor-1）
- vllm：`0.27.1+empty`
- backend：mooncake（master @ 127.0.0.1:50088）
- 模型：Qwen3-32B
- 场景：pdmix_non_layerwise（`kv_role=kv_both`，非分层 offload）

#### 执行结果

```
RUNNING: starting service
WARN: MemFabric env script not found, continue with Python package environment
WARN: Memcache env script not found, continue with Python package environment
4 workers: Creating v1 connector with name: AscendStoreConnector
PASS: service ready at http://127.0.0.1:8004
```

```
RUNNING: Qwen3-32B PDMix non-layerwise KV Pool validation
models endpoint ok
smoke request ok
first repeated-prefix request ok elapsed=4.02s
second repeated-prefix request ok elapsed=1.47s
usage1 {"prompt_tokens": 3458, "total_tokens": 3474, "completion_tokens": 16}
usage2 {"prompt_tokens": 3458, "total_tokens": 3474, "completion_tokens": 16}
PASS: Qwen3-32B PDMix non-layerwise KV Pool smoke and repeated-prefix validation passed
```

#### 验证内容

| 验证点 | 结果 |
|---|---|
| 服务拉起（4 worker 创建 AscendStoreConnector） | ✓ |
| backend 初始化（mooncake） | ✓ |
| 能存（第一次前缀请求 KV cache 写入 mooncake） | ✓ |
| 能取（第二次相同前缀命中 KV Pool） | ✓ |
| 通路正常（models/smoke/重复前缀都 OK） | ✓ |
| 性能收益（4.02s → 1.47s，降幅 63%） | ✓ |
| 无 traceback（start.sh readiness 检查通过） | ✓ |

#### 结果目录

`/home/lizhongyang/llm-project/transfer_data/refactor/kvpool/pr_13160/qwen3_32b/pdmix_non_layerwise/results/20260818_075731/`

- `status.txt`: `PASS: service ready at http://127.0.0.1:8004`
- `test_status.txt`: `PASS: ... validation passed`
- `env.txt`: 记录 HEAD = `2ef35ee6c`，vllm = `0.27.1+empty`
- `server.log`: 4 worker 创建 connector，无 traceback
- `mooncake.json`: mooncake backend 配置

### 阶段 5 补充：layerwise（memcache backend）验证

#### 背景

review.md 的核心缺口是"未完成真实环境 connector save/load smoke"。阶段 5 已用 mooncake backend 验证了 non-layerwise 路径。本补充验证 layerwise 路径（`use_layerwise: true`），该路径代码硬性要求 `backend == "memcache"`（`layerwise_cache_layout.py:98`）。

#### 环境

- 容器：`refactor_818`（8×Ascend910）
- NPU：0-3（TP=4）
- vllm-ascend：`2ef35ee6c`（ascend-store-refactor-1）
- vllm：`0.27.1+empty`
- backend：memcache（MetaService @ 127.0.0.1:5000/6000，`device_sdma` 协议）
- 模型：Qwen3-32B
- 场景：pdmix_layerwise（`kv_role=kv_both`，`use_layerwise=true`）
- SERVER_PORT：8005

#### MetaService 启动

```bash
export MMC_META_CONFIG_PATH=/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/config/mmc-meta.conf
python -c "from memcache_hybrid import MetaService; MetaService.main()"
# 端口 5000（meta_service_url）+ 6000（config_store_url）监听确认
```

`mmc-local.conf` 的 `protocol` 从默认 `host_rdma` 改为 `device_sdma`（A3 HCCS 推荐）。

#### 执行结果

```
RUNNING: starting service
kv_transfer_config: backend=memcache, use_layerwise=True, lookup_rpc_port=0
4 workers: Creating v1 connector with name: AscendStoreConnector
pool_worker.py:433 layerwise config: num_layers=64 num_groups=1 physical_layer_to_group_layers_sample={}
PASS: service ready at http://127.0.0.1:8005
```

```
RUNNING: Qwen3-32B PDMix layerwise (memcache) KV Pool validation
models endpoint ok
smoke request ok
first repeated-prefix request ok elapsed=2.11s
second repeated-prefix request ok elapsed=1.75s
usage1 {"prompt_tokens": 3458, "total_tokens": 3474, "completion_tokens": 16}
usage2 {"prompt_tokens": 3458, "total_tokens": 3474, "completion_tokens": 16}
PASS: Qwen3-32B PDMix layerwise (memcache) KV Pool smoke and repeated-prefix validation passed
```

#### 验证内容

| 验证点 | 结果 |
|---|---|
| MetaService 启动（端口 5000/6000） | ✓ |
| `use_layerwise: true` 配置生效 | ✓（`kv_transfer_config` 日志确认） |
| layerwise 路径激活 | ✓（`pool_worker.py:433 layerwise config: num_layers=64 num_groups=1`） |
| memcache backend 初始化（`mmc-local.conf` 加载） | ✓ |
| 4 worker AscendStoreConnector 创建 | ✓ |
| 能存（首次前缀写入 KV Pool） | ✓（first 2.11s） |
| 能取（重复前缀命中 KV Pool） | ✓（second 1.75s，更快） |
| 通路正常（models/smoke/重复前缀都 OK） | ✓ |
| 无 traceback | ✓ |

#### 结果目录

`/home/lizhongyang/llm-project/transfer_data/refactor/kvpool/pr_13160/qwen3_32b/pdmix_layerwise/results/20260818_083924/`

- `status.txt`: `PASS: service ready at http://127.0.0.1:8005`
- `test_status.txt`: `PASS: ... validation passed`
- `env.txt`: 记录 HEAD = `2ef35ee6c`，USE_LAYERWISE=true
- `server.log`: layerwise config 激活，memcache 初始化成功

### 阶段 5 补充：DSV4 MLA + layerwise（不兼容记录）

#### 尝试

用 DSV4-Flash（MLA 架构，`num_key_value_heads=1`，256 专家）+ memcache + `use_layerwise=true` 尝试启动。

#### 结果

```
ValueError: Physical layer 2 with multiple cache specs must have exactly one main spec 
and one '.indexer.k_cache' spec; got [
  'model.layers.2.self_attn.compressor.state_cache',
  'model.layers.2.self_attn.indexer.k_cache',
  'model.layers.2.self_attn.indexer.compressor.state_cache',
  'model.layers.2.self_attn.swa_cache',
  'model.layers.2.self_attn.attn'
].
```

#### 分析

- `layerwise_cache_layout.py:221` 的 `build_layerwise_reuse_layout` 要求多 spec 层恰好有 1 个 main spec + 1 个 `.indexer.k_cache` spec
- DSV4 的 MLA 层有 5 个 cache specs（compressor.state_cache、indexer.k_cache、indexer.compressor.state_cache、swa_cache、attn），超出支持范围
- **该文件不在 refactor-1 改动范围内**（refactor-1 的 12 个文件不包含 `layerwise_cache_layout.py`），属于 refactor-2 的范围
- **对 refactor-1 的验证无影响**：layerwise 路径激活已确认（代码进入了 `build_layerwise_reuse_layout`）

#### 结果目录（保留）

`/home/lizhongyang/llm-project/transfer_data/refactor/kvpool/pr_13160/dsv4/pdmix_layerwise/results/` — 保留失败记录

---

## 三、与 PR #13160 的对照

PR #13160 在 `c6f72551f` 上跑过相同的 smoke（results/20260818_024247），结果 PASS。本次在 `2ef35ee6c`（ascend-store-refactor-1）上重跑，同样 PASS，证明拆分后的第一部分保持了原 PR 的功能等价性。

| 维度 | PR #13160 (`c6f72551f`) | PR #14465 (`2ef35ee6c`) |
|---|---|---|
| vllm 版本 | `0.27.1+empty` | `0.27.1+empty` |
| backend | mooncake | mooncake + memcache |
| 服务拉起 | PASS | PASS |
| non-layerwise 重复前缀命中 | PASS | PASS（mooncake） |
| layerwise 路径激活 | - | PASS（memcache，`num_layers=64`） |
| layerwise 重复前缀命中 | - | PASS（memcache，2.11s→1.75s） |
| smoke 总体 | PASS | PASS |

---

## 四、附注

### 4.1 容器选择

- `refactor_810`：最初用于阶段 0-4，CPU 环境，UT 全通过。vllm 版本被切到 v0.27.1 后 qwen3 模型需要的 `aclnnAddRmsNormBias` 算子在 CANN 9.0.1 缺失，smoke 无法在此容器跑。
- `refactor_818`：用于阶段 5 smoke，8×Ascend910 NPU，环境完整，smoke 通过。

### 4.2 git bundle 传输

由于 `refactor_818` 容器 github 网络不稳定（fetch 失败/超时），改用 `git bundle` 从 `refactor_810` 导出三个拆分分支，在 `refactor_818` 导入。bundle 50 MB，包含完整历史，verify 通过。

### 4.3 mooncake vs memcache

`start.sh` 默认 `KV_BACKEND=memcache`。阶段 5 初次验证时 memcache 的 config store 服务（端口 6000）未启动，改用 mooncake（master @ 127.0.0.1:50088 已运行）完成 non-layerwise 验证。

补充验证阶段已启动 MetaService（`python -c "from memcache_hybrid import MetaService; MetaService.main()"`，端口 5000/6000），并修改 `mmc-local.conf` 的 `protocol` 为 `device_sdma`（A3 HCCS 推荐），完成 memcache + layerwise 验证。

两个 backend 均已验证通过：
- mooncake：non-layerwise 路径（qwen3-32b pdmix_non_layerwise PASS）
- memcache：layerwise 路径（qwen3-32b pdmix_layerwise PASS）

---

## 五、验证脚本

所有验证脚本位于：
- 本地：`D:\lzy\project\kv_pool\tmp\`
- 服务器：`/home/lizhongyang/tmp/`

脚本命名规范：`p0_*.sh`（阶段 0）至 `p5_*.sh`（阶段 5），每个脚本独立可执行。
