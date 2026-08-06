# PR 13354 backend 验证脚本

脚本目录：`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/backend/test_script`

目标 PR：`https://github.com/vllm-project/vllm-ascend/pull/13354`

## 推荐执行顺序

创建 run 目录并记录环境：

```bash
cd /home/lizhongyang/refactor/llm-project/transfer_data/refactor/backend/test_script
RUN_DIR=$(./00_env_snapshot.sh | awk -F= '/RUN_DIR=/{print $2}')
echo ${RUN_DIR}
```

结构验证：

```bash
python3 08_backend_structure_check.py > ${RUN_DIR}/backend_structure_check.json
```

导入、py_compile、目标 UT：

```bash
./01_check_import_ut.sh ${RUN_DIR}
```

更细分的目标测试：

```bash
cd /vllm-workspace/vllm-ascend
python3 -m pytest tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py -q
python3 -m pytest tests/ut/distributed/mooncake -q
```

如果 NPU 资源可用，再做服务级 smoke：

```bash
cd /home/lizhongyang/refactor/llm-project/transfer_data/refactor/backend/test_script
RUN_DIR=$(./00_env_snapshot.sh | awk -F= '/RUN_DIR=/{print $2}')
ASCEND_RT_VISIBLE_DEVICES=8 MODEL_PATH=/mnt/weight/Qwen3-0.6B PORT=8100 ./03_start_server_baseline.sh ${RUN_DIR}
python3 10_health_check.py --run-dir ${RUN_DIR} --port 8100
python3 05_send_requests.py --run-dir ${RUN_DIR} --case baseline --model /mnt/weight/Qwen3-0.6B --port 8100 --repeat 2
./07_stop_run.sh ${RUN_DIR}

RUN_DIR=$(./00_env_snapshot.sh | awk -F= '/RUN_DIR=/{print $2}')
MOONCAKE_MASTER_PORT=50088 ./02_start_mooncake_master.sh ${RUN_DIR}
ASCEND_RT_VISIBLE_DEVICES=8 MODEL_PATH=/mnt/weight/Qwen3-0.6B PORT=8100 ./04_start_server_kvpool_custom.sh ${RUN_DIR}
python3 10_health_check.py --run-dir ${RUN_DIR} --port 8100
python3 05_send_requests.py --run-dir ${RUN_DIR} --case kvpool_backend --model /mnt/weight/Qwen3-0.6B --port 8100 --repeat 2
./06_grep_logs.sh ${RUN_DIR}
./07_stop_run.sh ${RUN_DIR}
```

## 结果文件

每次 run 保存到：`runs/YYYYMMDD_HHMMSS_<case>/`

关键文件：

- `env.txt`：环境、commit、包路径、NPU 信息。
- `backend_structure_check.json`：backend 目录结构、旧路径缺失、新路径 import、connector registry、backend_map 检查。
- `check_import_ut.log`：required import、registry、py_compile、pytest 结果。
- `pytest_kv_transfer_failures.log`：KV transfer failure 相关目标测试。
- `pytest_mooncake.log`：Mooncake 相关目标测试。
- `path_reference_scan.txt`：新旧路径引用扫描。
- `resource_gate.txt`：是否适合启动服务级验证的资源快照。
- `conclusion.md`：人工结论。

## 当前注意点

- PR 13354 的 verified vLLM commit 是 `d02df748bf9efd99022f1a062597dc3cb3808485`。
- 本机 `/vllm-workspace/vllm` 当前没有该 commit 对象，GitHub fetch 之前超时；当前 vLLM 为 `568afb3a13806beb53bb2e6bd518269357b237c0`。
- `lmcache_ascend` 和 `ucm` 在当前环境未安装，所以这两个 connector 的实际 import 只记录为可选缺失。
- 当前 NPU 显示全卡高占用且对应宿主 PID 在容器内不可见，本轮没有启动服务级 smoke。服务脚本已归档，可在资源空闲后复跑。
