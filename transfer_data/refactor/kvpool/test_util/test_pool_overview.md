# AscendStore KV Pool 测试梳理

目标：先整理 `Simplify AscendStore KV pool Implementation` 这类改动后可以怎么测；本文件只梳理方案，不实际拉起服务。

参考文档：

- `code/vllm-ascend/docs/source/user_guide/feature_guide/kv_pool.md`
- `code/vllm-ascend/docs/source/user_guide/feature_guide/layerwise_kv_pool.md`

## 1. 优先测试顺序

建议按风险和成本从低到高测试：

1. 语法/导入级检查
2. AscendStore 单元测试
3. Mooncake 后端 PD-Mixed 单实例功能测试
4. Mooncake 后端 PD Disaggregation 双实例功能测试
5. Memcache 后端功能测试
6. Memcache layerwise 功能测试
7. 特殊场景回归：`load_async`、`consumer_is_to_put`、TP mismatch、hybrid/cache family

其中最适合作为第一轮人工验证的是 Mooncake PD-Mixed：只需要一个 vLLM 服务实例，直接用 `AscendStoreConnector` 的 `kv_both` 角色验证保存和二次命中。

## 2. 本地轻量检查

如果 Python 依赖齐全，先跑 AscendStore 相关 UT：

```bash
cd D:/lzy/project/kv_pool/code/vllm-ascend
python -m pytest tests/ut/distributed/ascend_store -q
```

如果没有 `pytest`，至少做语法检查：

```bash
cd D:/lzy/project/kv_pool/code/vllm-ascend
python -m py_compile \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/coordinator.py \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py \
  tests/ut/distributed/ascend_store/test_pool_worker.py
```

当前环境曾遇到的问题：`python -m pytest ...` 报 `No module named pytest`，所以真正 UT 需要在安装测试依赖的环境里执行。

## 3. Mooncake 后端准备

参考 `kv_pool.md`，Mooncake 后端需要：

- CANN / Ascend 环境可用。
- 安装 `mooncake-transfer-engine-npu`。
- 设置统一 hash：

```bash
export PYTHONHASHSEED=0
```

- 准备 `mooncake.json`，关键字段：

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "ascend",
  "device_name": "",
  "master_server_address": "<master_ip>:50088",
  "global_segment_size": "1GB",
  "preferred_segment": false,
  "prefer_alloc_in_same_node": true
}
```

- 设置配置路径：

```bash
export MOONCAKE_CONFIG_PATH=/path/to/mooncake.json
```

- 启动 Mooncake master：

```bash
mooncake_master --port 50088 \
  --eviction_high_watermark_ratio 0.9 \
  --eviction_ratio 0.1 \
  --default_kv_lease_ttl 11000
```

按硬件选择额外环境变量：

```bash
# A3 推荐
export ASCEND_ENABLE_USE_FABRIC_MEM=1

# A2 常见配置
# export HCCL_INTRA_ROCE_ENABLE=1

# A5 UBOE
# export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'

# A5 UB
# export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
```

建议也设置连接和传输超时：

```bash
export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export ACL_OP_INIT_MODE=1
```

## 4. Mooncake PD-Mixed 单实例测试

这是第一轮最推荐的端到端功能测试。

启动服务示例：

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/path/to/vllm
export MOONCAKE_CONFIG_PATH=/path/to/mooncake.json
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0
export ACL_OP_INIT_MODE=1
export ASCEND_ENABLE_USE_FABRIC_MEM=1
export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000

python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen2.5-7B-Instruct \
  --port 8100 \
  --trust-remote-code \
  --enforce-eager \
  --no-enable-prefix-caching \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 32768 \
  --block-size 128 \
  --max-num-batched-tokens 16384 \
  --kv-transfer-config '{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
      "lookup_rpc_port": "1",
      "backend": "mooncake"
    }
  }' > mix.log 2>&1
```

请求示例：

```bash
curl -s http://localhost:8100/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/Qwen2.5-7B-Instruct",
    "prompt": "Hello. I have a question. The president of the United States is",
    "max_completion_tokens": 32,
    "temperature": 0.0
  }'
```

建议验证方法：

1. 发送同一个长 prompt 第一次请求，让 KV 被保存到 pool。
2. 再发送相同 prefix 的第二次请求。
3. 观察日志里是否出现外部 KV Pool 命中、load/save、backend put/get 等信息。
4. 第二次请求应能走外部 KV 命中路径，TTFT 理论上应降低。

重点关注：

- 服务是否正常启动。
- `AscendStoreConnector` 是否注册成功。
- `MooncakeBackend` 是否初始化成功。
- 是否有 key lookup / put / get 日志。
- 第二次相同 prefix 是否命中。
- 请求是否正常返回，不触发 load failure 或 recompute 异常。

## 5. Mooncake PD Disaggregation 双实例测试

这个更接近生产 PD 分离场景，但启动成本更高。

文档里的结构是：

- Prefill 服务：`MultiConnector`，同时使用 `MooncakeConnectorV1` 和 `AscendStoreConnector`。
- Decode 服务：`MultiConnector`，同时使用 `MooncakeConnectorV1` 和 `AscendStoreConnector`。
- Proxy：`load_balance_proxy_server_example.py`。

Prefill 侧核心配置：

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_producer",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_producer",
        "kv_port": "20001",
        "kv_connector_extra_config": {
          "prefill": {"dp_size": 1, "tp_size": 1},
          "decode": {"dp_size": 1, "tp_size": 1}
        }
      },
      {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
          "lookup_rpc_port": "0",
          "backend": "mooncake"
        }
      }
    ]
  }
}
```

Decode 侧核心配置：

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_consumer",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_consumer",
        "kv_port": "20002",
        "kv_connector_extra_config": {
          "prefill": {"dp_size": 1, "tp_size": 1},
          "decode": {"dp_size": 1, "tp_size": 1}
        }
      },
      {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {
          "lookup_rpc_port": "0",
          "backend": "mooncake"
        }
      }
    ]
  }
}
```

Proxy 示例：

```bash
python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host localhost \
  --prefiller-hosts localhost \
  --prefiller-ports 8100 \
  --decoder-hosts localhost \
  --decoder-ports 8200
```

请求发到 proxy：

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/Qwen2.5-7B-Instruct",
    "prompt": "Given the accelerating impacts of climate change, explain the major challenges and possible global responses.",
    "max_completion_tokens": 128,
    "temperature": 0.0
  }'
```

重点验证：

- Prefill producer 能保存 KV 到 pool。
- Decode consumer 能查询并加载外部 KV。
- `kv_load_failure_policy: recompute` 下，load 失败时能回退重算而不是请求崩溃。
- `get_finished` / delayed free / async save 不导致 block 提前释放。

## 6. `consumer_is_to_put` 场景

文档提到 Decode 节点也可以把 KV Cache 写入 KV Pool，适用于 MLA 等场景。

核心配置：

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_consumer",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "lookup_rpc_port": "0",
    "backend": "mooncake",
    "consumer_is_to_put": true,
    "prefill_pp_size": 2,
    "prefill_pp_layer_partition": "30,31"
  }
}
```

测试关注：

- consumer 角色是否启动发送线程。
- decode 侧保存完成后是否正确上报 finished。
- prefill PP 分区配置是否参与 key 生成。
- 请求结束时 block 是否不会因为异步保存被提前释放。

## 7. Memcache 后端测试

参考 `kv_pool.md` 的 Memcache 部分，前置依赖：

```bash
pip install memfabric-hybrid
pip install memcache-hybrid
```

需要准备：

- `mmc-meta.conf`
- `mmc-local.conf`
- `MMC_META_CONFIG_PATH`
- `MMC_LOCAL_CONFIG_PATH`
- MetaService

MetaService 启动示例：

```bash
export MMC_META_CONFIG_PATH=/path/to/mmc-meta.conf
python -c "from memcache_hybrid import MetaService; MetaService.main()"
```

vLLM 配置里把 backend 改成：

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "lookup_rpc_port": "1",
    "backend": "memcache"
  }
}
```

重点验证：

- `MemcacheBackend` 初始化。
- `register_buffer` 成功。
- 普通 key I/O 路径 `exists/get/put` 正常。
- 如果启用 layerwise，则 GVA 相关 `batch_alloc/batch_get_key_info/batch_copy/batch_write_finish` 正常。

## 8. Memcache layerwise 测试

参考 `layerwise_kv_pool.md`。

前置：layerwise 当前要求 `backend: "memcache"`。

PD-Mixed 示例：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export PYTHONHASHSEED=0
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh

python -m vllm.entrypoints.openai.api_server \
  --model /path/to/DeepSeek-V2-Lite \
  --port 8100 \
  --trust-remote-code \
  --enforce-eager \
  --no-enable-prefix-caching \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --kv-transfer-config '{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "backend": "memcache",
      "lookup_rpc_port": "0",
      "use_layerwise": true
    }
  }'
```

注意：`layerwise_kv_pool.md` 里示例写的是 `mooncake_rpc_port`，而 `kv_pool.md` 参数表和 AscendStore 配置使用的是 `lookup_rpc_port`。实际测试时优先使用 `lookup_rpc_port`。

请求示例：

```bash
curl -s http://127.0.0.1:8100/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/DeepSeek-V2-Lite",
    "prompt": "Hello, my name is",
    "max_tokens": 32,
    "temperature": 0.0
  }'
```

重点验证：

- `use_layerwise=true` 时 piecewise/layerwise hook 是否正常。
- 每层 load/save 是否执行。
- `wait_for_layer_load` / `save_kv_layer` 是否被调用。
- `layerwise_prefetch_layers`、`layerwise_max_transfer_blocks`、`layerwise_max_transfer_bytes`、`h2d_stagger_us` 配置是否不会导致异常。

## 9. 建议的最小验收标准

第一轮可以先做到：

1. AscendStore 相关 UT 在有 pytest 的环境通过。
2. Mooncake PD-Mixed 单实例能启动。
3. 同一个长 prompt 连续请求两次，请求均成功返回。
4. 日志确认第二次请求出现外部 KV Pool 命中或 load 行为。
5. 没有明显异常：连接超时、backend 初始化失败、load failure 未处理、block 提前释放、线程未退出等。

如果第一轮通过，再扩展到：

- PD Disaggregation。
- Memcache backend。
- Layerwise。
- `consumer_is_to_put`。
- TP mismatch / hybrid / compressed cache family。

## 10. 这次 refactor 重点应覆盖的代码路径

`Simplify AscendStore KV pool Implementation` 涉及的测试重点建议映射如下：

| 风险点 | 建议测试 |
| --- | --- |
| scheduler 命中查询和 metadata 构造 | UT + PD-Mixed 二次相同 prefix 请求 |
| worker 注册 KV cache 和后端 buffer | 服务启动 + 首次请求保存 |
| backend exists/get/put | Mooncake 或 Memcache 实际后端测试 |
| async load failure 上报 | UT + 人工制造 backend get failure |
| delayed free / finished request | 长 prompt + 并发请求 + 请求结束后日志检查 |
| group block size / hybrid cache | 使用 hybrid 或 MLA 模型回归 |
| layerwise load/save | Memcache + `use_layerwise=true` |
| consumer_is_to_put | Decode consumer 配置 `consumer_is_to_put=true` |

## 11. 暂不建议第一轮测试的内容

这些场景成本较高，建议基础链路通过后再测：

- SSD offload。
- 多节点 PD disaggregation。
- 大规模 DP/TP/PP 组合。
- A5 UBOE / UB 专项网络配置。
- 长时间稳定性和性能压测。
