# PR 13354 验证脚本

脚本和记录都放在本目录下。

## 执行顺序
1. `./00_env_snapshot.sh`
2. `./01_check_import_ut.sh`
3. `./03_start_server_baseline.sh` + `./05_send_requests.py`
4. `./02_start_mooncake_master.sh` + `./10_start_server_kvpool_custom.sh` + `./05_send_requests.py`
5. `./09_send_stream_requests.py`
6. `./06_grep_logs.sh`
7. `./07_stop_run.sh`

## 记录文件
- `runs/<time>_<case>/env.txt`
- `runs/<time>_<case>/check_import_ut.log`
- `runs/<time>_<case>/server_baseline.log`
- `runs/<time>_<case>/server_kvpool_custom.log`
- `runs/<time>_<case>/mooncake_master.log`
- `runs/<time>_<case>/requests.jsonl`
- `runs/<time>_<case>/summary.json`
- `runs/<time>_<case>/stream_requests.jsonl`
- `runs/<time>_<case>/stream_summary.json`
- `runs/<time>_<case>/log_extract.txt`
