# 公共前置 — 池化服务启动前必做

> 所有场景启动 vllm 前的前置。memcache 路径需 §1–3；mooncake 路径需 §4；PD 双 connector 场景两者都要。
> 路径变量：`$BASE` = 服务器工作区（`/home/lizhongyang/map_XX`），`$CTR` = 容器名；以下按 165 实例写，换机替换。

## 1. hugepages（memcache device transfer 硬性前置）

缺了必须 **abort 不只是 WARN**——store 层 batch_copy 会返回 -1、请求全失败。

```bash
CUR=$(cat /proc/sys/vm/nr_hugepages)
[ "$CUR" -lt 200000 ] && sysctl -w vm.nr_hugepages=200000
grep -E 'HugePages_Total|HugePages_Free' /proc/meminfo   # 确认已分配
```

## 2. mmc-local.conf（memcache 本地服务配置，容器内）

site-packages 下 memcache_hybrid/config/mmc-local.conf，改两处：

```bash
CONF=$(python3 -c "import memcache_hybrid, os; print(os.path.join(os.path.dirname(memcache_hybrid.__file__), 'config'))")
cp -n $CONF/mmc-local.conf $CONF/mmc-local.conf.bak 2>/dev/null || true
sed -i 's/^ock.mmc.local_service.protocol = .*/ock.mmc.local_service.protocol = device_sdma/' $CONF/mmc-local.conf
sed -i 's/^ock.mmc.local_service.dram.size = .*/ock.mmc.local_service.dram.size = 16GB/' $CONF/mmc-local.conf
grep -E 'protocol|dram.size' $CONF/mmc-local.conf
```

- `protocol`：A3/HCCS → `device_sdma`；A2 或 RoCE 链路 → `device_rdma`（选型依据 verify_guide.md §1.4）
- `dram.size`：主机侧 DRAM 池容量（验证用 16GB，按机按需）

## 3. MetaService 拉起（memcache 等价 mooncake master，容器内）

```bash
export MMC_META_CONFIG_PATH=$CONF/mmc-meta.conf
pkill -f 'MetaService' 2>/dev/null; sleep 1
nohup python3 -c 'from memcache_hybrid import MetaService; MetaService.main()' \
  > $BASE/run/metaservice.log 2>&1 &
sleep 3
ss -tln | grep -E ':5000 |:6000 '   # 探活:5000/6000 监听才算就绪
curl -s http://127.0.0.1:8000/metrics | head -3   # Prometheus 指标端点("存"维独立证人源)
```

不拉起时 vllm init 连 6000 报 `errno:111`。

## 4. mooncake master 拉起（mooncake 路径，容器内）

mooncake.json（放 `$BASE/test/`）：

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

```bash
LOGD=$BASE/run/mooncake_logs; mkdir -p $LOGD
[ -f $LOGD/master.pid ] && kill -9 $(cat $LOGD/master.pid) 2>/dev/null; sleep 1   # 显式 PID,禁 pkill
mooncake_master --rpc_port 50088 --metrics_port 9008 > $LOGD/master.log 2>&1 &
echo $! > $LOGD/master.pid
sleep 3
curl -s http://127.0.0.1:9008/metrics | grep -E 'role|state|service_ready' | head -5
# 探活四查:role=leader / state=serving / service_ready=true / TCP 50088 通
# master 跨场景可复用;全部场景跑完再 kill -9 $(cat master.pid)
```

端口铁律：master RPC **50088**（50051 被宿主机 OceanStor DTMA 占用，51/112 实测）、metrics **9008**、
vllm API 错开。`--host` 一律绑 127.0.0.1，**禁 0.0.0.0**（共享服务器）。

## 5. 场景间清理（容器内，每场景结束必做）

```bash
# 停单实例场景:停 vllm(显式 PID → 等待 → kill -9 → 清 multiprocessing 残余)
bash $BASE/test/stop_server.sh $BASE/run/<场景目录>
# 停 PD 场景:proxy → decode → prefill 三个进程
bash $BASE/test/stop_s1.sh
# 清 NPU 残留 + 确认
bash $BASE/test/clean_npu.sh
```

stop 通用模式（自写脚本照此模板）：

```bash
PID=$(cat $DIR/<name>.pid)
kill $PID; for i in $(seq 1 10); do kill -0 $PID 2>/dev/null || break; sleep 2; done
kill -9 $PID 2>/dev/null
pkill -9 -f "from multiprocessing" 2>/dev/null   # vllm 崩溃残留 worker 占 HBM,必清
```

## 6. 一键前置（宿主机，memcache 路径）

```bash
bash $BASE/start/pool_prep.sh
# 输出 POOL_PREP_OK = hugepages + conf + MetaService 三步全过
```
