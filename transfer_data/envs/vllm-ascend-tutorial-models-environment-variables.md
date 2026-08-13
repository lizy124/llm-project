# `docs/source/tutorials/models` 环境变量清单

- 扫描目录：`D:/lzy/project/kv_pool/code/vllm-ascend/docs/source/tutorials/models`
- 文档文件数：**43**
- 明确环境变量语义的唯一名称：**54**
- 统计范围：模型教程中的 `export`、行内进程赋值、Docker `-e/--env`、Python 环境 API，以及正文明确说明为环境变量的名称。
- `IMAGE`、`MODEL_PATH`、`TAG`、`nic_name` 等命令辅助变量会保留，但单独分类，不视为产品配置。

## 分类统计

| 分类 | 变量数 | 占比 |
|---|---:|---:|
| vLLM Ascend 产品配置 | 6 | 11.1% |
| Ascend/CANN/HCCL 与 NPU 运行时 | 14 | 25.9% |
| 分布式通信与并行运行环境 | 4 | 7.4% |
| KV Transfer 与外部存储 | 1 | 1.9% |
| 上游 vLLM/PyTorch/模型生态 | 16 | 29.6% |
| 系统与通用运行环境 | 3 | 5.6% |
| 文档示例辅助变量 | 4 | 7.4% |
| 模型/评测专用变量 | 0 | 0.0% |
| 其他文档环境变量 | 6 | 11.1% |
| **合计** | **54** | **100.0%** |

## 使用形式统计

同一变量可出现多种使用形式，统计存在交集。

| 使用形式 | 变量数 |
|---|---:|
| export | 50 |
| inline assignment | 26 |
| Python environment API | 2 |
| 正文明确提及 | 2 |
| docker -e/--env | 1 |

## 覆盖最多的模型文档

| 文档 | 环境变量数 |
|---|---:|
| `GLM5.2.md` | 34 |
| `Qwen3.5-397B-A17B.md` | 23 |
| `MiniMax-M2.md` | 20 |
| `Qwen3-235B-A22B.md` | 14 |
| `Qwen3-VL-235B-A22B-Instruct.md` | 14 |
| `GLM5.md` | 13 |
| `Kimi-K2.6.md` | 11 |
| `Kimi-K2.5.md` | 11 |
| `DeepSeek-V3.2.md` | 11 |
| `gpt-oss-120b.md` | 11 |
| `DeepSeek-V3.1.md` | 10 |
| `Mixtral-8x7B-Instruct-v0.1.md` | 10 |
| `GLM4.x.md` | 9 |
| `Qwen3-VL-30B-A3B-Instruct.md` | 9 |
| `Qwen3-Omni-30B-A3B-Thinking.md` | 8 |
| `DeepSeek-R1.md` | 8 |
| `Qwen3-Coder-30B-A3B.md` | 6 |
| `Hunyuan-A13B-Instruct.md` | 6 |
| `Qwen3-30B-A3B.md` | 6 |
| `Qwen3-Dense.md` | 5 |

## 分类明细

位置使用 `docs/source/tutorials/models` 下的相对路径和行号；每个变量最多展示 8 个代表位置。

### vLLM Ascend 产品配置（6）

| 变量 | 使用形式 | 出现文档数 | 文档位置（示例） |
|---|---|---:|---|
| `VLLM_ASCEND_BALANCE_SCHEDULING` | export, inline assignment, 正文明确提及 | 9 | `docs/source/tutorials/models/DeepSeek-R1.md:143; docs/source/tutorials/models/DeepSeek-V3.1.md:154; docs/source/tutorials/models/GLM4.x.md:133; docs/source/tutorials/models/GLM5.2.md:188; docs/source/tutorials/models/GLM5.md:554; docs/source/tutorials/models/GLM5.md:603; docs/source/tutorials/models/Kimi-K2.5.md:147; docs/source/tutorials/models/Kimi-K2.5.md:277` |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | export, inline assignment | 11 | `docs/source/tutorials/models/DeepSeek-V3.1.md:453; docs/source/tutorials/models/DeepSeek-V3.1.md:528; docs/source/tutorials/models/DeepSeek-V3.2.md:136; docs/source/tutorials/models/DeepSeek-V3.2.md:189; docs/source/tutorials/models/DeepSeek-V3.2.md:236; docs/source/tutorials/models/DeepSeek-V3.2.md:427; docs/source/tutorials/models/DeepSeek-V3.2.md:500; docs/source/tutorials/models/DeepSeek-V4-Pro.md:317` |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | export, inline assignment | 4 | `docs/source/tutorials/models/GLM5.2.md:144; docs/source/tutorials/models/GLM5.2.md:221; docs/source/tutorials/models/GLM5.2.md:275; docs/source/tutorials/models/GLM5.2.md:601; docs/source/tutorials/models/GLM5.2.md:670; docs/source/tutorials/models/MiniMax-M2.md:299; docs/source/tutorials/models/MiniMax-M2.md:361; docs/source/tutorials/models/Qwen3-VL-235B-A22B-Instruct.md:183` |
| `VLLM_ASCEND_ENABLE_MLAPO` | export | 6 | `docs/source/tutorials/models/DeepSeek-V3.2.md:134; docs/source/tutorials/models/GLM5.2.md:1075; docs/source/tutorials/models/GLM5.md:555; docs/source/tutorials/models/GLM5.md:604; docs/source/tutorials/models/Kimi-K2.5.md:145; docs/source/tutorials/models/Kimi-K2.6.md:138; docs/source/tutorials/models/Mixtral-8x7B-Instruct-v0.1.md:79` |
| `VLLM_ASCEND_ENABLE_NZ` | export | 2 | `docs/source/tutorials/models/DeepSeekOCR2.md:127; docs/source/tutorials/models/GLM5.2.md:1258; docs/source/tutorials/models/GLM5.2.md:1324; docs/source/tutorials/models/GLM5.2.md:1390; docs/source/tutorials/models/GLM5.2.md:1463` |
| `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` | export, inline assignment | 1 | `docs/source/tutorials/models/GLM4.x.md:134; docs/source/tutorials/models/GLM4.x.md:184; docs/source/tutorials/models/GLM4.x.md:234` |

### Ascend/CANN/HCCL 与 NPU 运行时（14）

| 变量 | 使用形式 | 出现文档数 | 文档位置（示例） |
|---|---|---:|---|
| `ASCEND_AGGREGATE_ENABLE` | export | 1 | `docs/source/tutorials/models/GLM5.2.md:986` |
| `ASCEND_RT_VISIBLE_DEVICES` | docker -e/--env, export, inline assignment | 22 | `docs/source/tutorials/models/DeepSeek-V3.2.md:574; docs/source/tutorials/models/DeepSeek-V3.2.md:648; docs/source/tutorials/models/DeepSeek-V4-Flash.md:573; docs/source/tutorials/models/DeepSeek-V4-Flash.md:721; docs/source/tutorials/models/DeepSeek-V4-Flash.md:977; docs/source/tutorials/models/DeepSeek-V4-Pro.md:1042; docs/source/tutorials/models/DeepSeek-V4-Pro.md:737; docs/source/tutorials/models/DeepSeek-V4-Pro.md:966` |
| `ASCEND_TRANSPORT_PRINT` | export | 1 | `docs/source/tutorials/models/GLM5.2.md:987` |
| `CPU_AFFINITY_CONF` | export | 1 | `docs/source/tutorials/models/GLM5.2.md:873; docs/source/tutorials/models/GLM5.2.md:923` |
| `HCCL_BUFFSIZE` | export, inline assignment | 18 | `docs/source/tutorials/models/DeepSeek-V3.2.md:133; docs/source/tutorials/models/GLM4.x.md:128; docs/source/tutorials/models/GLM5.2.md:1076; docs/source/tutorials/models/GLM5.2.md:1262; docs/source/tutorials/models/GLM5.2.md:1328; docs/source/tutorials/models/GLM5.2.md:1394; docs/source/tutorials/models/GLM5.2.md:141; docs/source/tutorials/models/GLM5.2.md:1467` |
| `HCCL_CONNECT_TIMEOUT` | export | 1 | `docs/source/tutorials/models/GLM5.2.md:1265; docs/source/tutorials/models/GLM5.2.md:1331; docs/source/tutorials/models/GLM5.2.md:138; docs/source/tutorials/models/GLM5.2.md:1397; docs/source/tutorials/models/GLM5.2.md:1470; docs/source/tutorials/models/GLM5.2.md:215; docs/source/tutorials/models/GLM5.2.md:269; docs/source/tutorials/models/GLM5.2.md:867` |
| `HCCL_EXEC_TIMEOUT` | export | 1 | `docs/source/tutorials/models/GLM5.2.md:1264; docs/source/tutorials/models/GLM5.2.md:1330; docs/source/tutorials/models/GLM5.2.md:137; docs/source/tutorials/models/GLM5.2.md:1396; docs/source/tutorials/models/GLM5.2.md:1469; docs/source/tutorials/models/GLM5.2.md:214; docs/source/tutorials/models/GLM5.2.md:268; docs/source/tutorials/models/GLM5.2.md:866` |
| `HCCL_IF_IP` | export | 8 | `docs/source/tutorials/models/DeepSeek-R1.md:139; docs/source/tutorials/models/DeepSeek-V3.1.md:150; docs/source/tutorials/models/GLM5.2.md:1063; docs/source/tutorials/models/GLM5.2.md:1068; docs/source/tutorials/models/GLM5.2.md:1320; docs/source/tutorials/models/GLM5.2.md:1385; docs/source/tutorials/models/GLM5.2.md:1459; docs/source/tutorials/models/GLM5.2.md:209` |
| `HCCL_INTRA_PCIE_ENABLE` | export | 1 | `docs/source/tutorials/models/MiniMax-M2.md:225` |
| `HCCL_INTRA_ROCE_ENABLE` | export, inline assignment | 6 | `docs/source/tutorials/models/DeepSeek-R1.md:248; docs/source/tutorials/models/DeepSeek-R1.md:294; docs/source/tutorials/models/DeepSeek-V3.1.md:269; docs/source/tutorials/models/DeepSeek-V3.1.md:322; docs/source/tutorials/models/DeepSeek-V3.2.md:290; docs/source/tutorials/models/DeepSeek-V3.2.md:341; docs/source/tutorials/models/GLM5.2.md:1084; docs/source/tutorials/models/GLM5.2.md:991` |
| `HCCL_OP_EXPANSION_MODE` | export, inline assignment | 20 | `docs/source/tutorials/models/DeepSeek-R1.md:137; docs/source/tutorials/models/DeepSeek-V3.1.md:148; docs/source/tutorials/models/DeepSeek-V3.2.md:129; docs/source/tutorials/models/DeepSeek-V4-Flash.md:157; docs/source/tutorials/models/DeepSeek-V4-Flash.md:200; docs/source/tutorials/models/GLM4.x.md:132; docs/source/tutorials/models/GLM5.2.md:1078; docs/source/tutorials/models/GLM5.2.md:1259` |
| `HCCL_SOCKET_IFNAME` | export | 8 | `docs/source/tutorials/models/DeepSeek-R1.md:142; docs/source/tutorials/models/DeepSeek-V3.1.md:153; docs/source/tutorials/models/GLM5.2.md:1066; docs/source/tutorials/models/GLM5.2.md:1071; docs/source/tutorials/models/GLM5.2.md:1323; docs/source/tutorials/models/GLM5.2.md:1388; docs/source/tutorials/models/GLM5.2.md:1462; docs/source/tutorials/models/GLM5.2.md:212` |
| `HCCL_TRANSFER_TIMEOUT` | export | 1 | `docs/source/tutorials/models/GLM5.2.md:1263; docs/source/tutorials/models/GLM5.2.md:1329; docs/source/tutorials/models/GLM5.2.md:136; docs/source/tutorials/models/GLM5.2.md:1395; docs/source/tutorials/models/GLM5.2.md:1468; docs/source/tutorials/models/GLM5.2.md:213; docs/source/tutorials/models/GLM5.2.md:267` |
| `TASK_QUEUE_ENABLE` | export, inline assignment | 12 | `docs/source/tutorials/models/DeepSeek-V4-Flash.md:902; docs/source/tutorials/models/DeepSeekOCR2.md:130; docs/source/tutorials/models/GLM5.2.md:1077; docs/source/tutorials/models/GLM5.2.md:1268; docs/source/tutorials/models/GLM5.2.md:1334; docs/source/tutorials/models/GLM5.2.md:1400; docs/source/tutorials/models/GLM5.2.md:1473; docs/source/tutorials/models/GLM5.2.md:872` |

### 分布式通信与并行运行环境（4）

| 变量 | 使用形式 | 出现文档数 | 文档位置（示例） |
|---|---|---:|---|
| `GLOO_SOCKET_IFNAME` | export | 8 | `docs/source/tutorials/models/DeepSeek-R1.md:140; docs/source/tutorials/models/DeepSeek-V3.1.md:151; docs/source/tutorials/models/GLM5.2.md:1064; docs/source/tutorials/models/GLM5.2.md:1069; docs/source/tutorials/models/GLM5.2.md:1321; docs/source/tutorials/models/GLM5.2.md:1386; docs/source/tutorials/models/GLM5.2.md:1460; docs/source/tutorials/models/GLM5.2.md:210` |
| `OMP_NUM_THREADS` | export, inline assignment | 16 | `docs/source/tutorials/models/DeepSeek-V3.2.md:131; docs/source/tutorials/models/GLM4.x.md:130; docs/source/tutorials/models/GLM5.2.md:1073; docs/source/tutorials/models/GLM5.2.md:1261; docs/source/tutorials/models/GLM5.2.md:1327; docs/source/tutorials/models/GLM5.2.md:1393; docs/source/tutorials/models/GLM5.2.md:140; docs/source/tutorials/models/GLM5.2.md:1466` |
| `OMP_PROC_BIND` | export | 16 | `docs/source/tutorials/models/DeepSeek-V3.2.md:130; docs/source/tutorials/models/GLM4.x.md:129; docs/source/tutorials/models/GLM5.2.md:1072; docs/source/tutorials/models/GLM5.2.md:1260; docs/source/tutorials/models/GLM5.2.md:1326; docs/source/tutorials/models/GLM5.2.md:139; docs/source/tutorials/models/GLM5.2.md:1392; docs/source/tutorials/models/GLM5.2.md:1465` |
| `TP_SOCKET_IFNAME` | export | 8 | `docs/source/tutorials/models/DeepSeek-R1.md:141; docs/source/tutorials/models/DeepSeek-V3.1.md:152; docs/source/tutorials/models/GLM5.2.md:1065; docs/source/tutorials/models/GLM5.2.md:1070; docs/source/tutorials/models/GLM5.2.md:1322; docs/source/tutorials/models/GLM5.2.md:1387; docs/source/tutorials/models/GLM5.2.md:1461; docs/source/tutorials/models/GLM5.2.md:211` |

### KV Transfer 与外部存储（1）

| 变量 | 使用形式 | 出现文档数 | 文档位置（示例） |
|---|---|---:|---|
| `MOONCAKE_CONFIG_PATH` | export | 1 | `docs/source/tutorials/models/GLM5.2.md:1083; docs/source/tutorials/models/GLM5.2.md:990` |

### 上游 vLLM/PyTorch/模型生态（16）

| 变量 | 使用形式 | 出现文档数 | 文档位置（示例） |
|---|---|---:|---|
| `HF_DATASETS_CACHE` | Python environment API | 4 | `docs/source/tutorials/models/Qwen3-Embedding.md:222; docs/source/tutorials/models/Qwen3-Reranker.md:251; docs/source/tutorials/models/Qwen3-VL-Embedding.md:226; docs/source/tutorials/models/Qwen3-VL-Reranker.md:256` |
| `HF_ENDPOINT` | Python environment API | 4 | `docs/source/tutorials/models/Qwen3-Embedding.md:223; docs/source/tutorials/models/Qwen3-Reranker.md:252; docs/source/tutorials/models/Qwen3-VL-Embedding.md:227; docs/source/tutorials/models/Qwen3-VL-Reranker.md:257` |
| `HF_HOME` | export | 1 | `docs/source/tutorials/models/Hunyuan-A13B-Instruct.md:73` |
| `PYTORCH_NPU_ALLOC_CONF` | export, inline assignment | 20 | `docs/source/tutorials/models/DeepSeek-R1.md:144; docs/source/tutorials/models/DeepSeek-V3.1.md:155; docs/source/tutorials/models/DeepSeek-V3.2.md:135; docs/source/tutorials/models/DeepSeekOCR2.md:129; docs/source/tutorials/models/GLM4.x.md:131; docs/source/tutorials/models/GLM5.2.md:1074; docs/source/tutorials/models/GLM5.2.md:1266; docs/source/tutorials/models/GLM5.2.md:1332` |
| `TOKENIZERS_PARALLELISM` | export, inline assignment | 1 | `docs/source/tutorials/models/DeepSeekOCR2.md:128; docs/source/tutorials/models/DeepSeekOCR2.md:131` |
| `VLLM_ENGINE_READY_TIMEOUT_S` | export, inline assignment | 2 | `docs/source/tutorials/models/GLM5.2.md:874; docs/source/tutorials/models/GLM5.2.md:924; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:318; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:397; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:479` |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | export | 1 | `docs/source/tutorials/models/GLM5.2.md:865; docs/source/tutorials/models/GLM5.2.md:915` |
| `VLLM_HOST_IP` | export | 1 | `docs/source/tutorials/models/GLM5.2.md:1067` |
| `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` | export, inline assignment | 2 | `docs/source/tutorials/models/GLM5.2.md:1401; docs/source/tutorials/models/GLM5.2.md:1474; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:319; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:398; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:480` |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | inline assignment | 1 | `docs/source/tutorials/models/DeepSeek-V4-Flash.md:237; docs/source/tutorials/models/DeepSeek-V4-Flash.md:281; docs/source/tutorials/models/DeepSeek-V4-Flash.md:505; docs/source/tutorials/models/DeepSeek-V4-Flash.md:653` |
| `VLLM_RPC_TIMEOUT` | export, inline assignment | 2 | `docs/source/tutorials/models/DeepSeek-V4-Pro.md:169; docs/source/tutorials/models/DeepSeek-V4-Pro.md:245; docs/source/tutorials/models/GLM5.2.md:864; docs/source/tutorials/models/GLM5.2.md:914` |
| `VLLM_TARGET_DEVICE` | inline assignment | 2 | `docs/source/tutorials/models/Hunyuan-A13B-Instruct.md:50; docs/source/tutorials/models/Kimi-K2-Thinking.md:152` |
| `VLLM_TORCH_PROFILER_WITH_STACK` | export | 2 | `docs/source/tutorials/models/Qwen3-235B-A22B.md:410; docs/source/tutorials/models/Qwen3-235B-A22B.md:474; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:330; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:410; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:492` |
| `VLLM_USE_MODELSCOPE` | export, inline assignment | 10 | `docs/source/tutorials/models/DeepSeek-V3.2.md:831; docs/source/tutorials/models/MiniMax-M2.md:555; docs/source/tutorials/models/Qwen3-Next.md:225; docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md:393; docs/source/tutorials/models/Qwen3-VL-235B-A22B-Instruct.md:172; docs/source/tutorials/models/Qwen3-VL-235B-A22B-Instruct.md:233; docs/source/tutorials/models/Qwen3-VL-235B-A22B-Instruct.md:284; docs/source/tutorials/models/Qwen3-VL-235B-A22B-Instruct.md:373` |
| `VLLM_USE_V1` | export | 6 | `docs/source/tutorials/models/DeepSeek-V3.2.md:132; docs/source/tutorials/models/DeepSeekOCR2.md:126; docs/source/tutorials/models/GLM5.2.md:1079; docs/source/tutorials/models/GLM5.2.md:983; docs/source/tutorials/models/Mixtral-8x7B-Instruct-v0.1.md:77; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:326; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:406; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:488` |
| `VLLM_WORKER_MULTIPROC_METHOD` | export | 1 | `docs/source/tutorials/models/GLM5.2.md:1267; docs/source/tutorials/models/GLM5.2.md:1333; docs/source/tutorials/models/GLM5.2.md:1399; docs/source/tutorials/models/GLM5.2.md:1472` |

### 系统与通用运行环境（3）

| 变量 | 使用形式 | 出现文档数 | 文档位置（示例） |
|---|---|---:|---|
| `LD_LIBRARY_PATH` | export, inline assignment | 7 | `docs/source/tutorials/models/DeepSeek-V3.1.md:601; docs/source/tutorials/models/DeepSeek-V3.1.md:674; docs/source/tutorials/models/GLM4.x.md:396; docs/source/tutorials/models/GLM4.x.md:459; docs/source/tutorials/models/GLM5.2.md:1081; docs/source/tutorials/models/GLM5.2.md:985; docs/source/tutorials/models/GLM5.md:1066; docs/source/tutorials/models/GLM5.md:1135` |
| `LD_PRELOAD` | export, inline assignment | 7 | `docs/source/tutorials/models/Kimi-K2.5.md:132; docs/source/tutorials/models/Kimi-K2.6.md:142; docs/source/tutorials/models/MiniMax-M2.md:178; docs/source/tutorials/models/MiniMax-M2.md:217; docs/source/tutorials/models/MiniMax-M2.md:294; docs/source/tutorials/models/MiniMax-M2.md:356; docs/source/tutorials/models/MiniMax-M3.md:122; docs/source/tutorials/models/MiniMax-M3.md:155` |
| `PYTHONHASHSEED` | export | 2 | `docs/source/tutorials/models/GLM5.2.md:1082; docs/source/tutorials/models/GLM5.2.md:989; docs/source/tutorials/models/MiniMax-M2.md:300; docs/source/tutorials/models/MiniMax-M2.md:362` |

### 文档示例辅助变量（4）

| 变量 | 使用形式 | 出现文档数 | 文档位置（示例） |
|---|---|---:|---|
| `IMAGE` | export, inline assignment | 10 | `docs/source/tutorials/models/DeepSeek-V4-Flash.md:95; docs/source/tutorials/models/Hunyuan-A13B-Instruct.md:24; docs/source/tutorials/models/InternVL3.5.md:33; docs/source/tutorials/models/LLaVA-OneVision-Qwen2-0.5B-OV.md:30; docs/source/tutorials/models/Minitron-8B-Base.md:24; docs/source/tutorials/models/Mixtral-8x7B-Instruct-v0.1.md:38; docs/source/tutorials/models/Qwen2.5-Math-RM-72B.md:37; docs/source/tutorials/models/Qwen3-Next.md:39` |
| `MODEL` | export, inline assignment | 1 | `docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md:394` |
| `MODEL_PATH` | export, inline assignment, 正文明确提及 | 8 | `docs/source/tutorials/models/Gemma4.md:82; docs/source/tutorials/models/Gemma4.md:98; docs/source/tutorials/models/Hunyuan-A13B-Instruct.md:74; docs/source/tutorials/models/Hy3-preview.md:81; docs/source/tutorials/models/LLaVA-OneVision-Qwen2-0.5B-OV.md:57; docs/source/tutorials/models/PaddleOCR-VL.md:94; docs/source/tutorials/models/Qwen2.5-Math-RM-72B.md:68; docs/source/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md:211` |
| `NAME` | export, inline assignment | 6 | `docs/source/tutorials/models/Hy3-preview.md:35; docs/source/tutorials/models/InternVL3.5.md:34; docs/source/tutorials/models/Mixtral-8x7B-Instruct-v0.1.md:39; docs/source/tutorials/models/Qwen-VL-Dense.md:55; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:49; docs/source/tutorials/models/Qwen3.6-35B-A3B.md:40; docs/source/tutorials/models/Qwen3.6-35B-A3B.md:81` |

### 模型/评测专用变量（0）

| 变量 | 使用形式 | 出现文档数 | 文档位置（示例） |
|---|---|---:|---|

### 其他文档环境变量（6）

| 变量 | 使用形式 | 出现文档数 | 文档位置（示例） |
|---|---|---:|---|
| `ACL_OP_INIT_MODE` | export, inline assignment | 1 | `docs/source/tutorials/models/GLM5.2.md:1085; docs/source/tutorials/models/GLM5.2.md:871; docs/source/tutorials/models/GLM5.2.md:921; docs/source/tutorials/models/GLM5.2.md:992` |
| `IP_ADDRESS` | export | 1 | `docs/source/tutorials/models/Qwen3.5-397B-A17B.md:320; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:400; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:482` |
| `MASTER_IP_ADDRESS` | export | 1 | `docs/source/tutorials/models/Qwen3.5-397B-A17B.md:399; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:481` |
| `NETWORK_CARD_NAME` | export | 1 | `docs/source/tutorials/models/Qwen3.5-397B-A17B.md:321; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:401; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:483` |
| `NPU_MEMORY_FRACTION` | export | 1 | `docs/source/tutorials/models/gpt-oss-120b.md:104` |
| `TIKTOKEN_ENCODINGS_BASE` | export, inline assignment | 1 | `docs/source/tutorials/models/gpt-oss-120b.md:110; docs/source/tutorials/models/gpt-oss-120b.md:88` |

## 口径说明

1. 本文只覆盖 `docs/source/tutorials/models`，不扩展到源码、测试或其他文档目录。
2. 文档中的环境变量可能属于 vLLM Ascend、CANN/HCCL、PyTorch 或宿主系统；分类按主要用途组织。
3. 普通 Shell 局部赋值（如 `nic_name=...`、`MODEL_NAME=...`）不会因为被命令引用就自动算作环境变量；只有 `export`、命令前缀赋值等明确进入子进程环境的名称才纳入。
