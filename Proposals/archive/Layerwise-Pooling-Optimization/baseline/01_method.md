# HBM 基线组测试方法（前提数据摸底）

> 需求：SR20260820223202（见 [../01_requirements_analysis.md](../01_requirements_analysis.md)）
> 范围：仅 HBM PrefixCache 对照组 + 前提数据摸底；DRAM 池化组留待下轮
> 服务器：192.168.13.165（910×8，每卡 64G HBM）
> 权重：`/mnt/weight/DeepSeek-V4-Flash`（280G bf16，DeepseekV4ForCausalLM）
> 编制：2026-09-06

---

## 一、目标与产出

本轮不验证验收结论，只摸清后续实验依赖的前提数据：

| # | 产出 | 用途 |
|---|---|---|
| O1 | DSV4 Flash 实测 KV/token 字节数（hybrid c1/c4/c128 + MLA 无法纸面推导） | 容量规划：HBM 能缓存几个 128K 前缀 |
| O2 | HBM PrefixCache 可缓存满前缀数量（实测值） | 决定数据集前缀池规模 P |
| O3 | DSV4 Flash 128K 输入的 Prefill TPS 并发曲线（90% 命中下） | 验收基线天花板 + "最优并发"锚点 |
| O4 | 90% 命中负载构造方法验证（前缀灌入 + 命中率实测） | 后续 DRAM 池化组复用同一套负载 |
| O5 | TTFT 分布（mean/p99） | 需求分析补充的指标缺口基线值 |

## 二、环境

### 2.1 容器（按 playbook 流程新建）

- 镜像：`7f06feda13d3`（quay.nju.edu.cn/ascend/vllm-ascend nightly 系，2026-08-17）——refactor_165 同款底座，DSV4 支持已实证；165 上 7+ 容器在用
- 容器名：`baseline_165`
- 建容器脚本：[env/create_baseline_165.sh](env/create_baseline_165.sh)（playbook 标准模板 + refactor_165 实测挂载）
- 165 网络直连不受限，不走 CCW 代理

### 2.2 代码（不部署，用镜像预装）

镜像自带 `/vllm-workspace`（vllm + vllm-ascend 均 editable 安装指向源码树），核查通过（2026-09-06，check_env.sh）：

| 项 | 值 |
|---|---|
| vllm | `6e448d0`（0.27.1+empty，editable） |
| vllm-ascend | `33e849499`（0.19.1rc2.dev1561，editable） |
| 配对 | vllm-ascend 配对文件要求 v0.27.1 = 实装 0.27.1 ✓ |
| DSV4 | `DeepseekV4ForCausalLM` 在支持列表 ✓ |
| triton | 3.5.0 + triton_ascend 3.2.2（稳定配对） |
| benchmark | benchmark_serving.py 在 |

基线组只需标准 vllm 特性（prefix caching），不需要任何 PR 分支代码。
**纪律：`/vllm-workspace` 冻结**——禁止 git pull / checkout（editable 下动源码即换版本）；两包 commit 记入每轮归档。

### 2.3 服务器侧目录

```
/home/lizhongyang/lw_baseline/          ⇔ 本地 baseline/ 目录（scp 双向同步）
├── vllm/  vllm-ascend/                 # 代码（仅服务器侧）
├── data/                               # 合成数据集 + 生成脚本
├── run/                                # 启动与压测脚本、日志
└── results/<轮次名>/                   # 原始数据按轮归档，禁止覆盖
```

## 三、负载构造（O4 核心）

### 3.1 结构

```
请求 = 前缀（取自前缀池，可复用） + 后缀（每次随机生成）
```

| 参数 | 初值 | 定值依据 |
|---|---|---|
| 前缀长度 L_p | 115K token | (115K + 13K) = 128K 总输入，命中 115/128 ≈ 90% |
| 后缀长度 | 13K token | 同上 |
| 前缀池规模 P | 待 O1/O2 定（初值 64，**下界 32**） | 容量上界：P × L_p × KV/token ≤ HBM KV 可用容量的 ~70%（留运行余量）；多样性下界：P ≥ 32，保证最高并发档（128）时请求不至于集中撞同几个前缀 |
| 请求总数 R | 每轮 ≥ 10× 并发数 **且单轮时长 ≥ 10 分钟**（双条件取大） | 90% 命中下单请求实际只算 13K，完成很快；两条下界共同保证稳态采样充分 |

### 3.2 生成方式

- 前缀：随机 token id 序列（固定 seed，可复现）；后缀：每请求独立随机
- 输出格式对齐 `benchmark_serving.py` 支持的 custom 数据集格式（部署时以 `--help` 实际为准）
- token 数以 tokenizer 实测为准校验，目标命中率 90% ±1%

### 3.3 前缀灌入（预热）

- 压测前将 P 个前缀各发一次（output_len=1），全部完成后再开始计时压测
- 每轮并发扫描之间不重启服务（前缀保留在 HBM），但记录 `/metrics` 命中率确认未失效

## 四、指标定义

| 指标 | 口径 | 来源 |
|---|---|---|
| Prefill TPS | input token throughput（总输入 token / 压测时长） | benchmark_serving 输出 |
| TTFT | mean / p99 | benchmark_serving 输出 |
| 命中率 | gpu_prefix_cache_query_hit_rate（按 token） | vllm /metrics |
| KV/token | KV cache 总字节 ÷ (GPU blocks × block_size) | 启动日志 + block 数（hybrid 多组分别记录） |
| HBM 前缀容量 | 可用 KV 总量 ÷ 单前缀 KV 占用 | 由 KV/token 折算 + 实测灌入验证 |

## 五、服务配置（初版，首轮后修订）

| 参数 | 值 | 理由 |
|---|---|---|
| TP | 16 | 16 芯片（8 卡 × 2 芯，每芯 64G）；280G/16=17.5G 每芯权重，KV ~30G/芯。TP8 实测 KV 仅 14.75G/芯，装不下 160K 单请求（18.08G），已弃 |
| enable-prefix-caching | True | 基线组核心机制 |
| max-model-len | 160K | 128K 输入 + 输出余量 |
| gpu-memory-utilization | 0.9 | 惯例 |
| MTP / speculative | 关 | 纯净基线；特性叠加属后续轮次 |
| KV 传输 / 池化 | 全关 | 基线组无 DRAM 参与 |

## 六、执行步骤

```
S0  建容器 + 部署代码 + 端到端验证（playbook §8 清单）
S1  起服务（§五配置）→ 记录启动日志：GPU blocks、KV cache size、各组 block 数 → O1
S2  生成数据集（P 初值 64）→ 灌前缀 → 观察实际占用修正 P → O2
S3  并发扫描 4/8/16/32/64/128，每轮记录 TPS/TTFT/命中率 → O3/O5
S4  数据归档 results/，写 report.md
```

## 七、控制变量清单

| # | 变量 | 控制 |
|---|---|---|
| 1 | 前缀本地命中污染 | 基线组就是要本地命中（这是基线定义本身）；但每轮记录 /metrics 命中率，偏离 90%±1% 该轮作废 |
| 2 | 共享机干扰 | 跑压测前 npu-smi 截图存档；与其他容器高负载时段冲突时改期重跑 |
| 3 | 数据可复现 | 数据集 seed 固定；每轮归档完整 vllm serve 启动命令 + benchmark 命令 |
| 4 | 编译/预热抖动 | 服务起来后先发一轮 warmup 请求再开始正式压测 |
| 5 | 版本可追溯 | 归档 vllm/vllm-ascend 的 `git rev-parse HEAD` + pip show 版本串 |

## 八、本轮明确不做

- DRAM 池化组（AscendStore/memcache/mooncake 全不启用）
- DSV4 × layerwise 兼容性验证（P0 项另行推进，不阻塞本轮）
- SSD 路径、MTP/DCP/LayerSplit 叠加
- 极光大盘指标（属 AR20260820031213）
