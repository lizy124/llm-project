# 场景：单实例 + memcache + layerwise 冒烟

> 目的：layerwise 路径在 memcache 后端**非零激活**——排除 key 格式/协议层改动导致的
> 静默失效（永远 miss 且无报错）。
> 判据：`hit_tokens>0` + `valid_gvas>0` + MetaService 三维证据链（verify_guide.md §6.1④/§6.6）。
> 前置：[common-prerequisites.md](common-prerequisites.md) §1–3（hugepages + mmc conf + MetaService）。
> 默认拓扑：TP=4 单实例；模型选支持 layerwise 的 MLA/SFA 架构（DSV2-Lite 类；
> full attention 模型不集成 layerwise，选型依据 verify_guide.md §8.1 绕行方案）。

## 1. 环境变量设置（容器内）

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3            # 按空闲卡改
export MMC_LOCAL_CONFIG_PATH=$CONF/mmc-local.conf    # CONF 见 common §2
export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800             # TP=4 加载慢,给足
export VLLM_LOGGING_LEVEL=DEBUG                    # 判据要 grep DEBUG 行(load_gvas/hit_check)
export ACL_OP_INIT_MODE=1                           # A3/HCCS 分支;A2/RoCE 机器不设(见 verify_guide §1.4)
export ASCEND_ENABLE_USE_FABRIC_MEM=1               # A3/HCCS 分支(fabric mem)
```

## 2. 服务拉起

kv-transfer-config（JSON 传参，重点是 `"use_layerwise": true`）：

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "backend": "memcache",
    "lookup_rpc_port": "0",
    "use_layerwise": true
  }
}
```

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --host 0.0.0.0 --port 8004 \
  --served-model-name $SERVED_MODEL_NAME \
  --trust-remote-code --enforce-eager \
  --tensor-parallel-size 4 \
  --max-num-seqs 20 --max-model-len 32768 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 --block-size 128 \
  --no-enable-prefix-caching \
  --kv-transfer-config "$KV_CONFIG" \
  > server.log 2>&1 &
echo $! > server.pid
# 就绪等待:轮询 curl http://127.0.0.1:8004/v1/models 含 "data";进程退出先于就绪 = FAIL, tail server.log
```

参数要点：`--no-enable-prefix-caching` 必须（否则本地命中抑制池存取）；就绪超时 1800s。

## 3. 请求方式（python 内置 urllib，无需 client 库）

三段式：smoke → 共享长前缀首发（存入）→ 同前缀二发（命中）：

```python
import json, urllib.request
base = "http://127.0.0.1:8004"

def post(path, payload, timeout=300):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

# ① smoke 小请求(确认服务可用)
post("/v1/chat/completions", {"model": MODEL,
     "messages": [{"role": "user", "content": "Give one short sentence about prefix caching."}],
     "max_tokens": 32, "temperature": 0.0})

# ② 共享长前缀(~3300 token;prompt 需 ≥ 2 个完整 granularity 块,见 verify_guide §5.1)
shared = ("You are validating KV cache pool reuse. Keep this shared prefix identical "
          "between requests. The numbered facts are: 1 means alpha, 2 means beta, "
          "3 means gamma. Repeat these facts internally before answering. ") * 80

# ③ 首发存入 / ④ 二发命中(hit_check 判据来自这一发)
for q in ["Question A: answer with the word alpha only.",
          "Question B: answer with the word beta only."]:
    post("/v1/chat/completions", {"model": MODEL,
         "messages": [{"role": "user", "content": shared + q}],
         "max_tokens": 16, "temperature": 0.0})
```

关键：`temperature=0.0`、两次前缀**逐字节一致**——前缀不同则 key 不同、必然 miss。

## 4. 判定命令（正计数判据）

```bash
LOG=server.log
# ① layerwise 激活(配置面;注意这是无条件 info 日志,只证配置解析)
grep -E "layerwise config: num_layers" $LOG | tail -2
# ② 取:hit_tokens>0(排除 hit_tokens=0;smoke 小请求的 "no participating groups" 跳过属正常)
grep "hit_check:" $LOG | grep -v "hit_tokens=0" | tail -3
# ③ 存:valid_gvas>0(lease_fail=0 排除租约假阳性)
grep "load_gvas:" $LOG | grep -v "valid_gvas=0 " | tail -3
# ④ vllm metrics 跨层对拍(应与 hit_tokens 算术一致: keys × block_size)
curl -s http://127.0.0.1:8004/metrics | grep external_prefix_cache
# ⑤ MetaService 独立证人(存维,另一进程的计数)
curl -s http://127.0.0.1:8000/metrics | grep -E "^memcache_(alloc_successes|stored_keys|query_successes|query_not_found)"
# ⑥ 致命错误扫描
grep -Eiq 'Traceback|Segmentation fault|Store initialization failed' $LOG
```

通过形态示例：`hit_tokens=3328`、`valid_gvas=26 lease_fail=0`、
`alloc_successes=28 stored_keys=28 query_not_found=28`（miss 通道也在被观测，见 verify_guide §6.6③）。

## 5. 停止清理

```bash
bash $BASE/test/stop_server.sh $BASE/run/<本场景目录>
bash $BASE/test/clean_npu.sh
```

## 6. 已知坑

- memcache 1.2.0 无 put 类命名指标——"存"维用 `stored_keys` + `load_gvas` 双源交叉
- layerwise 数据面走 GVA 直读（device_sdma），不经 memcache get API——"取"维由
  `valid_gvas>0` + vllm external hits 承担，**别等 get 计数**
- 真 layerwise 路径标记是 `load_gvas:` / `hit_check:` DEBUG 行（pool_worker / pool_scheduler），
  "layerwise config" info 行不能当激活判据
- 与 [mooncake-non-layerwise.md](mooncake-non-layerwise.md) 构成对照对：**本场景 >0 且对照场景 = 0**
  才证明判定方法有效（verify_guide §6.6⑤）
