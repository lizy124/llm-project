# qwen_397b_pool P 节点启动失败定位记录

## 背景

目录：`D:\lzy\code\pd_pool_mtp\qwen_397b_pool`

该目录是在 `qwen_397b` 单机 16 卡 PD 分离部署基础上，加了池化能力的版本。部署形态仍然是：

- P / Prefill：NPU `0-7`，`4 DP x 2 TP`，HTTP 端口 `8000-8003`，vLLM DP RPC 端口 `12321`。
- D / Decode：NPU `8-15`，`4 DP x 2 TP`，HTTP 端口 `8004-8007`，vLLM DP RPC 端口 `12322`。
- P/D 通过 Mooncake 传 KV cache。
- 池化版在 `kv-transfer-config` 中使用 `MultiConnector`，包含 `MooncakeConnectorV1` 和 `AscendStoreConnector`。

当前现象：D 节点可以拉起，P 节点失败。主要查看日志文件：`D:\lzy\code\pd_pool_mtp\qwen_397b_pool\error.log`。

## 池化版 P/D 配置差异

P 模板：`run_dp_template_p.sh`

- `--data-parallel-rpc-port "${DP_RPC_PORT}"`，P 启动脚本传入 `12321`。
- `--max-model-len "${MAX_MODEL_LEN:-110000}"`。
- `kv-transfer-config` 使用：

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_producer",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "MooncakeConnectorV1",
        "kv_buffer_device": "npu",
        "kv_role": "kv_producer",
        "kv_port": "61001",
        "kv_connector_extra_config": {
          "prefill": {"dp_size": 4, "tp_size": 2},
          "decode": {"dp_size": 4, "tp_size": 2}
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

D 模板：`run_dp_template_d.sh`

- `--data-parallel-rpc-port "${DP_RPC_PORT}"`，D 启动脚本传入 `12322`。
- `--max-model-len "${MAX_MODEL_LEN:-140000}"`。
- 同样使用 `MultiConnector`，但 `kv_role` 是 `kv_consumer`，Mooncake `kv_port` 是 `61101`。

## 端口含义澄清

目前涉及多类端口，不能混在一起理解：

- `12321`：P 这一个 vLLM DP 组的 `--data-parallel-rpc-port`。
- `12322`：D 这一个 vLLM DP 组的 `--data-parallel-rpc-port`。
- `8000-8003`：P 各 DP rank 对外提供 OpenAI API 的 HTTP 端口。
- `8004-8007`：D 各 DP rank 对外提供 OpenAI API 的 HTTP 端口。
- `61001`：P 侧 Mooncake KV transfer 配置端口。
- `61101`：D 侧 Mooncake KV transfer 配置端口。
- `60000/60001`：日志里报错的 Ascend/HCCL 通信端口，不是 vLLM DP RPC 端口，也不是 Mooncake `61001/61101`。

`60000/60001` 在当前 `vllm` / `vllm-ascend` Python 代码和 `qwen_397b_pool` 脚本里没有显式默认值。它大概率来自 Ascend CANN / HCCL runtime 的默认通信端口段。

## DP 组说明

这里的一个 DP 组是指：一批使用同一套 `--data-parallel-size`、同一个 `--data-parallel-address`、同一个 `--data-parallel-rpc-port`，并且 `--data-parallel-rank` 覆盖 `0..DP_SIZE-1` 的 vLLM 实例。

P 的 4 个 rank 构成一个 Prefill DP 组：

```bash
bash run_dp_template_p.sh 0,1 8000 4 0 90.90.97.27 12321 2
bash run_dp_template_p.sh 2,3 8001 4 1 90.90.97.27 12321 2
bash run_dp_template_p.sh 4,5 8002 4 2 90.90.97.27 12321 2
bash run_dp_template_p.sh 6,7 8003 4 3 90.90.97.27 12321 2
```

D 的 4 个 rank 构成另一个 Decode DP 组：

```bash
bash run_dp_template_d.sh 8,9   8004 4 0 90.90.97.27 12322 2
bash run_dp_template_d.sh 10,11 8005 4 1 90.90.97.27 12322 2
bash run_dp_template_d.sh 12,13 8006 4 2 90.90.97.27 12322 2
bash run_dp_template_d.sh 14,15 8007 4 3 90.90.97.27 12322 2
```

因此 P 组内部共用 `12321` 是合理的，D 组内部共用 `12322` 是合理的，但 P/D 是两套独立 DP 组，不能全部写成 `12321`。

## 日志中的表象错误

`error.log` 前段出现：

```text
EngineCore failed to start.
RuntimeError: Did not receive response from front-end process within 5 minutes
```

典型位置：

```text
EngineCore_DP3 ... RuntimeError: Did not receive response from front-end process within 5 minutes
EngineCore_DP1 ... RuntimeError: Did not receive response from front-end process within 5 minutes
```

这个表象说明 EngineCore 向前端发起 startup handshake 后，5 分钟内没有收到前端返回的 init message。

但这不是最终根因。后续日志显示 P 侧 worker 已经开始加载模型和初始化 connector，只是启动链路太慢并且后面发生更底层异常。

## 日志中的关键根因 1：Host pinned memory 申请失败

真正关键错误之一：

```text
torch.OutOfMemoryError: allocate_host_memory_slowpath:
../torch_npu/csrc/core/npu/CachingHostAllocator.cpp:252
NPU function error: aclrtMallocHostWithCfg, error code is 207001
```

随后 Ascend runtime 报：

```text
ERR00100 PTA call acl api failed
Failed to apply for memory.
Memory_Allocation_Failure(EL0004): Failed to allocate memory requested by RUNTIME module.
rtsMallocHost execution failed, reason=driver error:out of memory
```

这里失败的不是普通 Python 堆内存，也不是模型权重显存本身，而是 Ascend/NPU runtime 申请的 host pinned memory，即主机侧锁页内存/页锁定内存。

具体失败点包括：

1. `Worker_DP3_TP1_EP7` 在初始化 `NPUInputBatch` 时失败：

```text
initialize_kv_cache
-> may_reinitialize_input_batch
-> NPUInputBatch(...)
-> self.repetition_penalties_cpu_tensor = torch.empty(...)
-> aclrtMallocHostWithCfg failed
```

2. `Worker_DP3_TP0_EP6` 在初始化 `MultiGroupBlockTable` 时失败：

```text
initialize_kv_cache
-> may_reinitialize_input_batch
-> NPUInputBatch(...)
-> MultiGroupBlockTable(...)
-> BlockTable(...)
-> CpuGpuBuffer(..., pin_memory=...)
-> torch.zeros(...)
-> aclrtMallocHostWithCfg failed
```

池化版日志中还能看到每个 worker 会挂载约 10GiB segment：

```text
Mounting segment: 10737418240 bytes, 10737418240 of 10737418240
```

这会显著增加 host 侧内存/锁页内存压力。

## 日志中的关键根因 2：HCCL 默认端口被占用

另一个关键错误：

```text
Communication_Error_Bind_IP_Port(EI0019): Failed to enable listening for the host network adapter socket.
Reason: The IP address 90.90.97.27 and port 60000 have already been bound.
```

后续还出现：

```text
Failed to enable listening for the host network adapter socket. Reason: The IP address 90.90.97.27 and port 60001 have already been bound.
```

这说明 Ascend/HCCL 通信初始化时尝试绑定 `90.90.97.27:60000` 和 `90.90.97.27:60001`，但端口已经被其他进程占用。

这些端口不是：

- 不是 vLLM DP RPC 端口 `12321/12322`。
- 不是 Mooncake KV transfer 端口 `61001/61101`。
- 不是 HTTP 服务端口 `8000-8007`。

它们属于 Ascend/HCCL 通信端口段，通常可以通过环境变量 `HCCL_IF_BASE_PORT` 显式调整。

## 当前判断

P 节点失败不是单一的 `12321/12322` 配错问题。

当前更可信的定位是：

1. 池化版 P 引入 `MultiConnector + AscendStoreConnector` 后，启动过程中需要更多 host pinned memory 和额外通信初始化。
2. P 侧启动耗时较长，表面出现 vLLM startup handshake 5 分钟超时。
3. 更底层实际异常是 Ascend runtime host pinned memory 申请失败。
4. 同时 HCCL 默认通信端口 `60000/60001` 已被占用，说明 P/D 或残留进程之间存在 HCCL 端口段冲突。

## 建议处理顺序

### 1. 先清理残留进程和端口占用

在目标机器上确认是否还有旧的 P/D/vLLM/HCCL 进程占用端口或 NPU 资源。重点检查：

- `60000/60001` 是否被旧进程占用。
- `8000-8007` 是否有旧 vLLM 服务。
- `12321/12322` 是否有旧 DP 组残留。
- 是否有旧的 Ascend runtime 进程没有退出干净。

### 2. 显式错开 P/D 的 HCCL 端口段

建议在 P/D 模板里设置不同的 `HCCL_IF_BASE_PORT`。

例如 P 使用默认段：

```bash
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-60000}"
```

D 使用另一个段：

```bash
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-61000}"
```

如果 D 已经占用了 `60000`，则让 P 使用：

```bash
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-62000}"
```

关键是 P 和 D 两套独立 DP 组不要同时落在同一个 HCCL 默认端口段。

### 3. 降低 P 池化版 host pinned memory 压力

建议先用更保守参数验证能否启动：

```bash
MAX_NUM_SEQS=4
MAX_MODEL_LEN=80000
GPU_MEMORY_UTILIZATION=0.85
```

当前 P 模板默认是：

```bash
--max-num-seqs "${MAX_NUM_SEQS:-10}"
--max-model-len "${MAX_MODEL_LEN:-110000}"
--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}"
```

池化开启后，`max_num_seqs`、block table、input batch、KV cache 配置都会影响 CPU pinned memory / host memory 申请量。

### 4. 不要优先修改 12321/12322

`12321/12322` 分别对应 P/D 的 vLLM DP RPC 端口。当前日志中的 `60000/60001` 冲突不属于这两个端口。

P/D 是两套独立 DP 组，P 保持 `12321`，D 保持 `12322` 是合理的。不要为了修 `60000/60001` 把 `12321/12322` 改成一样。

### 5. 注意 VLLM_STARTUP_TIMEOUT 不一定有效

P 模板里有：

```bash
export VLLM_STARTUP_TIMEOUT=6000
```

但当前报错仍显示：

```text
Did not receive response from front-end process within 5 minutes
```

这说明这个环境变量没有覆盖当前 vLLM startup handshake 的 5 分钟限制，或者只影响了其他启动等待逻辑。即使增大启动超时，也只是避免慢启动误判，不能解决 host pinned memory OOM 和 HCCL 端口占用。

## 一句话结论

池化版 P 失败的主因是：开启 `MultiConnector + AscendStoreConnector` 后，P 侧初始化需要更多 host pinned memory，并触发 `aclrtMallocHostWithCfg` 分配失败；同时 HCCL 默认端口 `60000/60001` 被占用，说明 P/D 或残留进程的 Ascend/HCCL 通信端口段没有隔离。优先清理残留进程、显式设置不同的 `HCCL_IF_BASE_PORT`，再降低 P 的 `MAX_NUM_SEQS` / `MAX_MODEL_LEN` / `GPU_MEMORY_UTILIZATION` 做启动验证。
