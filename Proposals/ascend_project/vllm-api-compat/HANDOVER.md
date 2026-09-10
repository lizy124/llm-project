HANDOVER.md
# 交接文档：vllm-api-compat 全量参数兼容测试（map_135）

> 交接时间：2026-09-07 17:00 (UTC+8)
> 最后更新：2026-09-07 18:40 (UTC+8)——Qwen3.5-4B 补跑完成，全流程收尾
> 前置会话：已完成主线全量测试 + SPEC 真实权重补跑 + 结果分支保存
> 新会话任务：~~等 Qwen3.5-4B 下载完成后，补跑 LoRA/MM/Exception 三个 section，结果追加 commit~~ **已完成**（commit b5a5099ba）

---

## 1. 一句话现状

**全部完成。** 200 用例最终口径 **178 PASS / 22 FAIL（零 Ascend 特有回归）**，含 Qwen3.5-4B 真实权重补跑（LoRA 8/8 + MM 9/9 + calculate-kv-scales 转正，共 18 例转 PASS）。结果已 commit 到 vllm-ascend 分支 `api-compat-20260907`（5634d3082 全量 + b5a5099ba 补跑，本地 commit 未 push）。22 个 FAIL 全部为非 Ascend 原因（vLLM 参数依赖校验 10、方法学/资产限制 4、HBM 竞争残留 8），详见测试报告 §3/§4。

## 2. 铁律（最高优先级）

1. **绝不 `git push`**，不上传公网——所有 commit 仅限本地
2. 不提 PR（用户会自己提）
3. 所有远程命令的输出必须重定向保留到日志文件

## 3. 环境与访问

| 项 | 值 |
|---|---|
| 服务器 | 80.5.9.135（node-97-36），ssh root 免密 |
| 容器 | pr15367_135（在容器内跑测试） |
| 工作目录（容器内） | /home/lizhongyang/map_135/api_compat |
| vllm | 0.27.1+empty，editable → /home/lizhongyang/code/vllm |
| vllm-ascend | 0.1.dev4937+g03e0e41e1，editable → /home/lizhongyang/code/vllm-ascend（分支 api-compat-20260907，基于 refactor_layerwise_part1/PR15367） |
| skill 位置 | 容器内 /home/lizhongyang/map_135/api_compat/.agents/skills/vllm-api-compat/（本地镜像 D:\project\code\vllm-ascend-workspace） |
| 本地脚本库 | D:\project\agent_project\map_135\start\（所有 push 过去的 .sh 都在这里有源文件） |

### ⚠️ 关键坑：双 /mnt/weight（必读）

- **宿主机**（`ssh root@80.5.9.135 "..."` 直连）的 /mnt/weight = **NFS**（172.27.1.31:/weight，80 个模型）
- **容器内**（go.ps1 map_135 或 docker exec）的 /mnt/weight = **宿主机本地盘**（bind mount 不传递 NFS 子挂载，目前 7 项资产，见下表）
- **同一路径两个视角内容不同**，排查前先分清自己在哪个视角

### 容器内 /mnt/weight 现有资产

| 资产 | 大小 | 状态 |
|---|---|---|
| Qwen3-0.6B | - | ✅ 完整（主力测试模型） |
| Qwen3-32B-W8A8 | 40G | ✅ 完整（SPEC 用） |
| EAGLE3-Qwen3-32B | 3G | ✅ 完整（SPEC draft） |
| qwen35-4b-text-only-sql-lora | - | ✅ 完整（LoRA adapter，modelscope clone） |
| **Qwen3.5-4B** | 8.8G | ✅ **完整**（curl 分片 + tokenizer 3 文件 LFS 指针已直链补下） |
| DeepSeek-V2-Lite-Chat / Qwen3-235B-A22B-w8a8-rot | - | 原有，与本任务无关 |

磁盘：本地盘剩 ~48G（99% used），**勿再放大文件**。

## 4. 下载监控（已结束，存档）

后台进程：curl 分片下载（容器内 PID 可能为 296424/296425 那组，若已退出看日志尾部有无 DOWNLOAD_DONE）

```bash
# 检查下载进度（容器内）
docker exec pr15367_135 du -sh /mnt/weight/Qwen3.5-4B
# 下载日志（宿主机视角）
cat /home/lizhongyang/map_135/api_compat/run_logs/qwen35_shards_download.log
# 完成标志：日志尾部出现 DOWNLOAD_DONE，且目录下无 .tmp 文件、两个 safetensors 共 ~8.8G
docker exec pr15367_135 ls -la /mnt/weight/Qwen3.5-4B/
```

**注意**：原 git clone 只拉到 LFS 指针（容器无 git-lfs），真实权重是 curl 直链下的两个分片：
- model.safetensors-00001-of-00002.safetensors（5.3G）
- model.safetensors-00002-of-00002.safetensors（3.8G）
- 其余小文件（config.json/tokenizer 等）git clone 时已齐

**若下载失败**：参考 D:\project\agent_project\map_135\start\download_qwen35_shards.sh 重启（走代理 http://80.254.14.6:3128，modelscope 直链 `https://www.modelscope.cn/models/Qwen/Qwen3.5-4B/resolve/master/<file>`）。curl 支持 `-C -` 断点续传。

## 5. Qwen3.5-4B 补跑任务（已完成，结果见报告 §4.7）

### 5.1 LoRAConfig ×8 → **8/8 PASS** ✅

adapter `/mnt/weight/qwen35-4b-text-only-sql-lora` + 基座 Qwen3.5-4B 就位后，全部 LoRA 参数（mixed-moe-lora-format/specialize-active-lora/fully-sharded-loras 及 no- 变体）加载成功。补跑 yaml/脚本：D:\project\agent_project\map_135\start\params_qwen35_rerun.yaml + run_qwen35_rerun.sh。

### 5.2 MultiModalConfig ×7 → **9/9 PASS** ✅

Qwen3.5-4B 多模态基线 + 内联 base64 图片请求全部正常返回。

### 5.3 ExceptionError → 终值确认 ✅

| 参数 | 结果 |
|---|---|
| --calculate-kv-scales | **PASS（转正）**，原 FAIL 实为基线路径 /home/weights/Qwen3.5-4B 不存在 |
| --enable-eplb / --enable-elastic-ep / --ray-workers-use-nsight | FAIL（本次补跑确认，pydantic 参数依赖校验，非 Ascend） |
| --grpc | FAIL（缺 smg-grpc-servicer pip 包，沿用旧记录） |
| --headless | FAIL（方法学限制：无前端进程握手超时，沿用旧记录） |
| --compilation-config.use_inductor_graph_partition | PASS（selective rerun 已转正） |

### 5.4 结果追加 commit → **已完成** ✅

commit b5a5099ba（9 文件：补跑 stdout ×6 + summary_upserted/status/test_report 更新），本地 commit 未 push。本次遇到的坑：PowerShell→go.ps1 链路中 commit message 含括号会导致引号解析失败，改用无括号消息。

### 5.5 补跑关键修复（存档）

- Qwen3.5-4B 的 tokenizer.json/vocab.json/merges.txt 是 **git-lfs 指针**（133B），vllm 加载 tokenizer 报错 → modelscope 直链补下真实文件（12.8MB/6.7MB/3.3MB）
- health_wait 对"启动即崩"用例干等满 timeout（grpc 卡 900s）→ 短 timeout 60s 重跑快速校验类用例

## 6. 已完成工作速查

| 阶段 | 结果 | 产物 |
|---|---|---|
| 并行全量 20 sections | 197 用例 157/40 | run_logs/full_parallel_stdout.json |
| DataParallelLB 串行 | 1 PASS / 2 FAIL（Non-MoE 校验） | run_logs/dplb_stdout.json |
| 选择性重跑（偶发嫌疑） | 41 result 34 PASS / 7 FAIL(=MM×7) | run_logs/selective_stdout.json |
| SPEC 真实权重（32B+EAGLE3） | **4/4 PASS** | run_logs/spec32b_stdout.json |
| parallel_drafting（32B 配对） | FAIL：draft 无 pard_token（vLLM 上游校验 llm_base_proposer.py:362） | run_logs/pd32b_stdout.json |
| **Qwen3.5-4B 补跑（阶段 5）** | **LoRA 8/8 + MM 9/9 + calculate-kv-scales 转 PASS（18 例转正）** | run_logs/loraconfig_35_stdout.json 等 6 个 + qwen35_rerun_watch.log |
| 12 个误报 fix stub | 全部无效（is_ascend_error 误报），已归档不外提 | .vaws-local/vllm-api-compat/daily/fixes/ |

**最终口径：200 用例 178 PASS / 22 FAIL，零 Ascend 特有回归。** 22 FAIL 全部归因：vLLM 参数依赖校验 10（ParallelConfig/ExceptionError/MOE/DataParallelLB）、方法学/资产限制 4（grpc/headless/parallel_drafting/compilation-config 残留旧记录）、并行 HBM 竞争残留 8（ObservabilityConfig 4 + PROFILER 4）。

## 7. 关键文档

| 文档 | 位置 |
|---|---|
| **完整测试报告**（含全部根因分析、skill 误报分析、环境架构） | D:\project\llm-project\ascend_project\vllm-api-compat\test_report_135_20260907.md |
| 服务器侧报告+status 副本 | 容器 /home/lizhongyang/map_135/api_compat/（同名） |
| 本地证据镜像（全量+DPLB+selective+spec+pd32b+阶段5 补跑 + summary_upserted/status.txt） | D:\project\agent_project\map_135\results\latest\ |
| skill 本地源（yaml 参数矩阵） | D:\project\code\vllm-ascend-workspace\.agents\skills\vllm-api-compat\ |

## 8. 常用命令备忘

```powershell
# 容器内执行命令（自动处理多层引号转义）
powershell -NoProfile -ExecutionPolicy Bypass -File D:\project\agent_project\_config\go.ps1 map_135 "<bash命令>"

# 宿主机执行
ssh root@80.5.9.135 "<命令>"

# 容器内执行
ssh root@80.5.9.135 "docker exec pr15367_135 bash -c '<命令>'"

# 传文件到服务器
scp <本地文件> root@80.5.9.135:/home/lizhongyang/map_135/api_compat/
```

**引号坑**：PowerShell → ssh → bash 多层嵌套时，含引号/awk/sed/管道的复杂命令必须走 go.ps1（base64 转义）或先写成 .sh push 过去执行；多行 git commit message 也要走脚本文件。

## 9. skill 改进建议（已写入报告 §5，待用户反馈给 workspace 仓库）

1. `wait_for_ready` 增加 `proc.poll()` 早退（启动即崩的用例别干等 400s）
2. baseline 依赖资产做 pre-flight 存在性检查（LoRA/多模态/draft 模型缺失直接 SKIP）
3. `is_ascend_error` 排除 pydantic ValidationError 类确定性文案（本次 12 个误报 stub 的来源）
4. `--rerun-failed` 支持只重跑"根因未识别"子集