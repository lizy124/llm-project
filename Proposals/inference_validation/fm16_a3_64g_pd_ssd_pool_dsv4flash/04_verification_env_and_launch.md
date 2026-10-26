
# 04 环境准备、启动与停止

本章按执行顺序说明完整过程；场景配置与脚本、通用工具的仓库引用和参数在 [03](03_verification_scripts.md)。本次整理没有登录服务器、初始化磁盘或重新运行测试。当前机器状态在执行前重新检查。

## 1. 固定起点与参数

采用 P `80.48.29.133/wl_1011`、D `80.48.29.134/wl_1011_new` 的对应已安装环境，依赖与缺口按 [02](02_development_and_pr.md) 核对。以下均为 Linux Bash 命令，注明宿主或容器；不要直接粘贴到本地 PowerShell。

| 项目 | P | D |
| --- | --- | --- |
| 可见 die | 0–15 | 0–15 |
| DP/TP | 2/8 | 8/2 |
| 单 rank 分组 | DP0=0–7；DP1=8–15 | DPi=2i、2i+1，i=0..7 |
| API | 8000、8001 | 8101–8108 |
| DP RPC | 12345 | 13345 |
| PD KV 配置 | 30000、30001 | 30200；保持来源连接器处理方式，改变端口策略须复验 |
| HCCL 基端口 | 51000 | 51000 |
| 通信域 | superpod=2、server_index=0、net_instance=2 | superpod=2、server_index=1、net_instance=2 |
| 网卡 | enp48s3u1u1 | enp48s3u1u1 |
| Meta / config store / metrics | 5000 / 6000 / 8004 | 连接 P |
| 代理 | 8080 | 无 |

库路径、通信域和模型选项来自来源环境；迁移到其他镜像或节点时联动核对。HBM 规格以实际 `npu-smi` 清单为准，区分物理卡与 die。模型参数 `max-model-len=134144`、P batched tokens=4096、D=192、block size=32、P/D 显存比例 0.96/0.95 均随 03 保存。

## 2. 导出文件、准备独立运行目录

在本地把 03 的每个 `file:` 代码块保存到对应相对路径，或使用 03 开头的 PowerShell 导出工具，得到 `03_verification_scripts/config/`、`launch/`、`tests/`。通用 DP launcher 从远端已有的 vllm-ascend 代码仓引用，不在文档导出物中重复提供。运行文件在独立临时目录展开，本地归档目录保持五份 Markdown 文档。按仓库远程传输纪律，经内网 scp 将这些文件放到两端的新运行根目录，原目录不覆盖。

两端宿主机分别设置：

```bash
TASK_ROOT=/home/lizhongyang/inference_validation/fm16_a3_64g_pd_ssd_pool_dsv4flash
mkdir -p "$TASK_ROOT/ssd_round/logs"
```

确认该目录可被对应容器以同一路径访问；若容器未挂载此路径，使用原环境已挂载的任务专属目录，或在补齐容器创建信息后重建专用容器。不要假设宿主新建目录会自动出现在容器内。

在各自容器内：

```bash
export TASK_ROOT=/home/lizhongyang/inference_validation/fm16_a3_64g_pd_ssd_pool_dsv4flash
cd "$TASK_ROOT"
mkdir -p ssd_round/logs
test -r /mnt/share/weight/DeepSeek-V4-Flash-DSpark-w4a8-scope-ci/config.json
test -r /usr/local/Ascend/cann/set_env.sh
python3 -c 'import memcache_hybrid, memfabric_hybrid, torch, torch_npu, vllm, vllm_ascend'
```

在两端容器中确认可访问的工具代码仓路径和版本：

```bash
export VLLM_ASCEND_REPO=/home/lizhongyang/code/vllm-ascend
test -r "$VLLM_ASCEND_REPO/examples/external_online_dp/launch_online_dp.py"
git -C "$VLLM_ASCEND_REPO" rev-parse HEAD
```

与 03 中固定的工具引用版本核对；路径不同时修改 `VLLM_ASCEND_REPO`。只读取该仓库中的启动工具，不为此切换容器内已安装的 vLLM/vllm-ascend。若接口不同，先对照 03 的参数说明，不沿用旧定制 launcher 的选项。

在 P 容器取得原代理实现并保留原件：

```bash
test -s /home/w00985460/script/vllm/load_balance_proxy_server_example.py
test ! -e "$TASK_ROOT/03_verification_scripts/launch/load_balance_proxy_server_example.py"
cp /home/w00985460/script/vllm/load_balance_proxy_server_example.py "$TASK_ROOT/03_verification_scripts/launch/"
sha256sum "$TASK_ROOT/03_verification_scripts/launch/load_balance_proxy_server_example.py"
```

这段首次准备命令应逐条确认，任一步失败就停止。原件缺失时流程缺少必要依赖，先从该原环境取得，不能任取同名代理替代。

## 3. 设备、端口、大页与通信预检

两端宿主机执行：

```bash
npu-smi info
ip address show enp48s3u1u1
ss -lntp
grep -E 'MemAvailable|HugePages_Total|HugePages_Free|HugePages_Rsvd|Hugepagesize' /proc/meminfo
cat /proc/sys/vm/nr_hugepages
docker ps -a
```

只使用本次明确空闲的全部所需 die；不要停止他人任务来凑设备。端口检查包括上表和 HCCL 派生端口。P/D 互访、HCCS 和框架基本算子须可用后再拉大模型。

大页按 standalone 容量与所用版本的实际要求准备。先保存原 `vm.nr_hugepages`，结合实际 Hugepagesize、空闲大页和宿主可用内存计算增量，再在宿主调整并核对；不自动照抄历史 200000 页。分配失败或空闲页不足时停止排查，退出后按本次归属恢复，不能释放其他任务使用的页。

## 4. 准备 SSD 层测试设备

本来源使用每节点独立的 200GiB 后备文件和 `/dev/loop0`。loop 是 UBS IO 使用的测试块设备；只有核实后备文件实际位于 SSD 文件系统上，才可声明物理介质也是 SSD。只验证后端 SSD 层功能时也要明确这个区别。

两端宿主机分别先查看：

```bash
findmnt -T /weights
df -h /weights
lsblk -o NAME,PATH,TYPE,SIZE,FSTYPE,MOUNTPOINTS
losetup -a
```

首次准备可使用下面独立文件路径。已存在的文件不能执行重新定长或初始化；先核对其归属和复用方式。

```bash
SSD_IMG=/weights/kv_pool_ssd/inference_validation_fm16.img
mkdir -p /weights/kv_pool_ssd
```

仅在确认该文件不存在后创建：

```bash
( set -eC; : > "$SSD_IMG"; truncate -s 200G "$SSD_IMG" )
```

首次关联前确认 `/dev/loop0` 没有被占用：

```bash
if losetup /dev/loop0 >/dev/null 2>&1; then
  echo 'loop0 已被关联：先核对归属，不执行新的绑定。'
else
  losetup /dev/loop0 "$SSD_IMG"
fi
losetup -l /dev/loop0
lsblk -o NAME,PATH,TYPE,SIZE,MOUNTPOINTS /dev/loop0
```

**继续的条件是** loop0 的后备文件确实为本任务文件、设备无挂载点。被其他文件占用时不要解绑或初始化；选择空闲 loop 并在该节点的 standalone 配置同步改 `ubsio.disk.path`。已有本任务映射可以复用，但同样要检查后备路径。

各自容器内检查该设备节点。若宿主已存在 loop0，而容器不可见，在已确认的对应容器中补同号块节点：

```bash
test -b /dev/loop0 || mknod /dev/loop0 b 7 0
ls -l /dev/loop0
```

使用其他 loop 号时按宿主的实际 major/minor 设置，不能仍用 7:0。最后核对 03 配置中的路径、每机 DRAM>0 的 LocalService 数为 1、首次 `force_new_disk=true` 的初始化影响。一次正常的保存/命中测试期间不重启 standalone；需要保留缓存时使用经核对的 false 配置。

## 5. 逐个拉起服务

以下在各自容器内执行；每个服务脚本前台运行，通过外层 `nohup` 放后台。先设置 TASK_ROOT、切到运行根目录并确认日志目录存在。每次运行使用新的 `LOG_DIR`，PID 文件也位于该目录。

```bash
export TASK_ROOT=/home/lizhongyang/inference_validation/fm16_a3_64g_pd_ssd_pool_dsv4flash
export LOG_DIR="$TASK_ROOT/ssd_round/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
cd "$TASK_ROOT"
```

P 容器先启动 Meta：

```bash
nohup bash 03_verification_scripts/launch/meta_ssd.sh > "$LOG_DIR/meta.launch.log" 2>&1 &
echo $! > "$LOG_DIR/meta.pid"
curl --noproxy '*' -fsS http://80.48.29.133:8004/metrics > /dev/null
```

Meta 未就绪时检查其日志，等待就绪后再继续。来源 meta/standalone 脚本还会追加到 `$TASK_ROOT/ssd_round/logs/` 中的固定文件，首次启动前保存旧文件；不能仅看旧日志中的初始化成功字样。

P、D 容器各启动一个 standalone：

```bash
nohup bash 03_verification_scripts/launch/standalone_ssd.sh > "$LOG_DIR/standalone.launch.log" 2>&1 &
echo $! > "$LOG_DIR/standalone.pid"
```

核对本次进程仍存活、本次日志出现初始化成功、Meta 中 DRAM/SSD 段和磁盘状态正常后，再启动 P/D。容量出现只说明注册完成。

P 容器先准备独立模板目录，再调用仓库 launcher。仓库工具固定读取当前目录的 `run_dp_template.sh`，因此将本场景 P 脚本链接为该名称；不改动代码仓模板：

```bash
mkdir -p "$TASK_ROOT/runtime/p"
cd "$TASK_ROOT/runtime/p"
ln -s "$TASK_ROOT/03_verification_scripts/launch/p_rank.sh" run_dp_template.sh
nohup python3 "$VLLM_ASCEND_REPO/examples/external_online_dp/launch_online_dp.py" \
  --dp-size 2 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 \
  --dp-address 80.48.29.133 --dp-rpc-port 12345 --vllm-start-port 8000 \
  > "$LOG_DIR/p_launcher.log" 2>&1 &
echo $! > "$LOG_DIR/p_launcher.pid"
```

D 容器使用自己的模板目录和 D 脚本：

```bash
mkdir -p "$TASK_ROOT/runtime/d"
cd "$TASK_ROOT/runtime/d"
ln -s "$TASK_ROOT/03_verification_scripts/launch/d_rank.sh" run_dp_template.sh
nohup python3 "$VLLM_ASCEND_REPO/examples/external_online_dp/launch_online_dp.py" \
  --dp-size 8 --tp-size 2 --dp-size-local 8 --dp-rank-start 0 \
  --dp-address 80.48.29.134 --dp-rpc-port 13345 --vllm-start-port 8101 \
  > "$LOG_DIR/d_launcher.log" 2>&1 &
echo $! > "$LOG_DIR/d_launcher.pid"
```

首次创建链接应逐条确认；链接已存在时先检查 `readlink -f run_dp_template.sh` 是否指向本场景的正确脚本，再跳过创建，不能忽略链接错误继续启动。这里的 launcher 没有逐 rank 日志和错峰启动选项，所有 rank 输出进入本机 launcher 日志；替换后尚未在目标环境复验。它不代替单 rank 脚本里的显式拓扑，两个位置必须一致。整体退出状态也不等于全部 rank 成功，继续前检查所有 API。

在任意可访问两节点的终端检查：

```bash
for port in 8000 8001; do
  curl --noproxy '*' -fsS "http://80.48.29.133:$port/health" || break
done
for port in 8101 8102 8103 8104 8105 8106 8107 8108; do
  curl --noproxy '*' -fsS "http://80.48.29.134:$port/health" || break
done
curl --noproxy '*' -fsS http://80.48.29.133:8000/v1/models
```

两个循环全部通过且模型名为 `ds` 后，P 容器启动代理：

```bash
cd "$TASK_ROOT"
nohup bash 03_verification_scripts/launch/proxy.sh > "$LOG_DIR/proxy.log" 2>&1 &
echo $! > "$LOG_DIR/proxy.pid"
curl --noproxy '*' -fsS http://80.48.29.133:8080/healthcheck
```

## 6. 准备客户端并进行简单测试

P 容器内准备独立的测试目录。AISBench 与前缀工具由单独整理的工具包提供，部署到本次专属的 `$AISBENCH_TOOL_DIR`；本流程不复制或维护其内部实现。把 03 的必要配置放为该工具目录的 `config.py`，按工具包要求准备数据。先核对工具版本、数据来源和环境，缺依赖时不能继续。

```bash
export TEST_PY=/mnt/share/l30041859/venv/ais_bench/bin/python
export AISBENCH_TOOL_DIR="$TASK_ROOT/03_verification_scripts/tests/aisbench"
"$TEST_PY" -c 'import numpy, pyarrow, torch, transformers, pandas, requests, ais_bench; print(numpy.__version__, pyarrow.__version__)'
test -s "$AISBENCH_TOOL_DIR/aisbench_test.py"
```

`config.py` 的 WORK_PATH 必须为本次测试专属、可写的 AISBench 安装/源码根目录。工具会改写模型配置和 GSM8K 链接，测试前备份这些位置；禁止与他人测试共用可写目录。旧虚拟环境路径可以作为起点，但它引用的安装目录也必须先确认归属。

按 [05](05_verification_process_and_results.md) 依次执行代理短请求、DRAM 前缀检查、SSD 多前缀检查。128K/1 的 SSD 场景准备如下：

```bash
RUN_DIR="$TASK_ROOT/checks/$(date +%Y%m%d_%H%M%S)_ssd"
mkdir -p "$RUN_DIR/logs"
curl --noproxy '*' -fsS http://80.48.29.133:8004/metrics > "$RUN_DIR/tier.before.prom"
export INPUT_LEN=131072 OUTPUT_LEN=1 CONCURRENCY=32 SEED=201 PREFIX_NUM=32 DATA_NUM=64
cd "$AISBENCH_TOOL_DIR"
```

执行 [03 第 2 节](03_verification_scripts.md) 的唯一 AISBench 测试命令，结束后继续读取分层差值：

```bash
curl --noproxy '*' -fsS http://80.48.29.133:8004/metrics > "$RUN_DIR/tier.after.prom"
python3 "$TASK_ROOT/03_verification_scripts/tests/tier_delta.py" "$RUN_DIR/tier.before.prom" "$RUN_DIR/tier.after.prom"
```

这是工作集触发方式，不保证任何一次都命中 SSD。确认 warmup 与正式阶段的实际成功数、输出长度和分层增量；具体判据见 05。这些临时快照服务于本次检查，无需将测试成绩填入本套文档。

## 7. 停止与恢复

停止顺序：客户端 → 代理 → P/D rank 与 worker → standalone → Meta。先用 PID 文件和当前进程树确认本轮归属，对具体进程发送 TERM 并等待退出，再复查设备和端口。launcher 的 PID 不包含全部 worker，不能只停止 launcher 就认为设备已释放。

```bash
ps -eo pid,ppid,lstart,args --forest
npu-smi info
ss -lntp
```

只对经过上述核对的本轮 PID 操作；不要使用来源编排中的广泛 `pkill`、全局共享内存删除或为此停止他人服务。若使用了本轮专门创建的独占容器，也可从宿主停止该专用容器；现有 `wl_1011`/`wl_1011_new` 不因名称出现在文档中就自动视为本轮独占。

全部相关进程退出后，按本次调整记录恢复大页。loop 设备只有在确属本轮新建且已无人使用时才可解绑，后备文件保留。恢复测试工具改动过的 AISBench 配置与数据链接。保留原任务目录全部内容。