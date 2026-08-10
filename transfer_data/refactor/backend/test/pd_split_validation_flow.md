# PR 13354 PD 分离补充验证

## 目标
验证 Mooncake + PD 分离场景下，`kv_producer` / `kv_consumer` 配置可启动、可路由、可完成请求。

## 资源选择
- 使用后面的空闲 NPU 卡
- 避开当前已有进程占用的前几张卡

## 启动顺序
1. `./00_env_snapshot.sh <run_dir>`
2. `./02_start_mooncake_master.sh <run_dir>`
3. `./11_start_server_pd_prefill.sh <run_dir>`
4. `./12_start_server_pd_decode.sh <run_dir>`
5. `./13_start_pd_proxy.sh <run_dir>`
6. 通过 proxy 发送请求
7. `./06_grep_logs.sh <run_dir>`
8. `./07_stop_run.sh <run_dir>`

## 记录文件
- `env.txt`
- `mooncake_master.log`
- `server_pd_prefill.log`
- `server_pd_decode.log`
- `pd_proxy.log`
- `requests.jsonl`
- `summary.json`
- `log_extract.txt`

## 关注点
- `kv_producer` / `kv_consumer` 是否都能正常启动
- proxy 是否能把请求路由到两端
- 日志里是否有 `ImportError`、`ModuleNotFoundError`、`Address already in use`、`RPC_FAIL`、`mooncake_backend` 初始化异常
