# `docs/source/tutorials/models` 环境变量清单

- 扫描目录：`D:/lzy/project/kv_pool/code/vllm-ascend/docs/source/tutorials/models`
- 文档文件数：**43**
- 明确环境变量语义的唯一名称：**58**
- 统计范围：模型教程中的 `export`、行内进程赋值、Docker `-e/--env`、Python 环境 API，以及正文明确说明为环境变量的名称。
- `IMAGE`、`MODEL_PATH`、`TAG` 等已导出的命令辅助变量会保留，但单独分类，不视为产品配置；`nic_name` 等未导出的 Shell 局部变量不纳入。

## 分类统计

| 分类 | 变量数 | 占比 |
|---|---:|---:|
| vLLM Ascend 产品配置 | 6 | 10.3% |
| Ascend/CANN/HCCL 与 NPU 运行时 | 17 | 29.3% |
| 分布式通信与并行运行环境 | 4 | 6.9% |
| KV Transfer 与外部存储 | 1 | 1.7% |
| 上游 vLLM/PyTorch/模型生态 | 16 | 27.6% |
| 系统与通用运行环境 | 3 | 5.2% |
| 文档示例辅助变量 | 4 | 6.9% |
| 模型/评测专用变量 | 0 | 0.0% |
| 其他文档环境变量 | 7 | 12.1% |
| **合计** | **58** | **100.0%** |

## 使用形式统计

同一变量可出现多种使用形式，统计存在交集。

| 使用形式 | 变量数 |
|---|---:|
| export | 55 |
| inline assignment | 26 |
| Python environment API | 2 |
| 正文明确提及 | 2 |
| docker -e/--env | 1 |

## 覆盖最多的模型文档

| 文档 | 环境变量数 |
|---|---:|
| `GLM5.2.md` | 37 |
| `DeepSeek-V4-Pro.md` | 24 |
| `DeepSeek-V3.2.md` | 24 |
| `GLM5.md` | 23 |
| `Qwen3.5-397B-A17B.md` | 23 |
| `DeepSeek-V3.1.md` | 22 |
| `GLM4.x.md` | 21 |
| `Kimi-K2.5.md` | 21 |
| `Kimi-K2.6.md` | 21 |
| `MiniMax-M2.md` | 21 |
| `MiniMax-M3.md` | 18 |
| `DeepSeek-V4-Flash.md` | 18 |
| `Qwen3.5-27B-Qwen3.6-27B.md` | 16 |
| `Qwen3-235B-A22B.md` | 15 |
| `Qwen3-VL-235B-A22B-Instruct.md` | 15 |
| `DeepSeek-R1.md` | 13 |
| `InternVL3.5.md` | 12 |
| `gpt-oss-120b.md` | 11 |
| `Qwen3.6-35B-A3B.md` | 10 |
| `Qwen3-VL-30B-A3B-Instruct.md` | 10 |

## 分类明细

位置使用 `docs/source/tutorials/models` 下的相对路径和行号；每个变量最多展示 8 个代表位置。

### vLLM Ascend 产品配置（6）

| 变量 | 使用形式 | 出现文档数 | 文档位置（示例） |
|---|---|---:|---|
| `VLLM_ASCEND_BALANCE_SCHEDULING` | export, inline assignment, 正文明确提及 | 9 | `docs/source/tutorials/models/DeepSeek-R1.md:143; docs/source/tutorials/models/DeepSeek-R1.md:246; docs/source/tutorials/models/DeepSeek-R1.md:292; docs/source/tutorials/models/DeepSeek-V3.1.md:154; docs/source/tutorials/models/DeepSeek-V3.1.md:267; docs/source/tutorials/models/DeepSeek-V3.1.md:320; docs/source/tutorials/models/GLM4.x.md:133; docs/source/tutorials/models/GLM4.x.md:183` |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | export, inline assignment | 13 | `docs/source/tutorials/models/DeepSeek-V3.1.md:453; docs/source/tutorials/models/DeepSeek-V3.1.md:528; docs/source/tutorials/models/DeepSeek-V3.2.md:136; docs/source/tutorials/models/DeepSeek-V3.2.md:189; docs/source/tutorials/models/DeepSeek-V3.2.md:236; docs/source/tutorials/models/DeepSeek-V3.2.md:287; docs/source/tutorials/models/DeepSeek-V3.2.md:338; docs/source/tutorials/models/DeepSeek-V3.2.md:427` |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | export, inline assignment | 8 | `docs/source/tutorials/models/DeepSeek-V4-Pro.md:596; docs/source/tutorials/models/DeepSeek-V4-Pro.md:667; docs/source/tutorials/models/GLM4.x.md:524; docs/source/tutorials/models/GLM4.x.md:593; docs/source/tutorials/models/GLM5.2.md:144; docs/source/tutorials/models/GLM5.2.md:221; docs/source/tutorials/models/GLM5.2.md:275; docs/source/tutorials/models/GLM5.2.md:446` |
| `VLLM_ASCEND_ENABLE_MLAPO` | export | 6 | `docs/source/tutorials/models/DeepSeek-V3.2.md:134; docs/source/tutorials/models/DeepSeek-V3.2.md:187; docs/source/tutorials/models/DeepSeek-V3.2.md:234; docs/source/tutorials/models/DeepSeek-V3.2.md:285; docs/source/tutorials/models/DeepSeek-V3.2.md:336; docs/source/tutorials/models/GLM5.2.md:1075; docs/source/tutorials/models/GLM5.md:1065; docs/source/tutorials/models/GLM5.md:1134` |
| `VLLM_ASCEND_ENABLE_NZ` | export | 2 | `docs/source/tutorials/models/DeepSeekOCR2.md:127; docs/source/tutorials/models/GLM5.2.md:1258; docs/source/tutorials/models/GLM5.2.md:1324; docs/source/tutorials/models/GLM5.2.md:1390; docs/source/tutorials/models/GLM5.2.md:1463` |
| `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` | export, inline assignment | 1 | `docs/source/tutorials/models/GLM4.x.md:134; docs/source/tutorials/models/GLM4.x.md:184; docs/source/tutorials/models/GLM4.x.md:234; docs/source/tutorials/models/GLM4.x.md:394; docs/source/tutorials/models/GLM4.x.md:457; docs/source/tutorials/models/GLM4.x.md:523; docs/source/tutorials/models/GLM4.x.md:592` |

### Ascend/CANN/HCCL 与 NPU 运行时（17）

| 变量 | 使用形式 | 出现文档数 | 文档位置（示例） |
|---|---|---:|---|
| `ASCEND_A3_ENABLE` | export | 4 | `docs/source/tutorials/models/DeepSeek-V3.2.md:421; docs/source/tutorials/models/DeepSeek-V3.2.md:494; docs/source/tutorials/models/DeepSeek-V3.2.md:568; docs/source/tutorials/models/DeepSeek-V3.2.md:642; docs/source/tutorials/models/GLM4.x.md:393; docs/source/tutorials/models/GLM4.x.md:456; docs/source/tutorials/models/GLM4.x.md:518; docs/source/tutorials/models/GLM4.x.md:587` |
| `ASCEND_AGGREGATE_ENABLE` | export | 4 | `docs/source/tutorials/models/DeepSeek-V3.2.md:418; docs/source/tutorials/models/DeepSeek-V3.2.md:491; docs/source/tutorials/models/DeepSeek-V3.2.md:565; docs/source/tutorials/models/DeepSeek-V3.2.md:639; docs/source/tutorials/models/GLM4.x.md:390; docs/source/tutorials/models/GLM4.x.md:453; docs/source/tutorials/models/GLM4.x.md:515; docs/source/tutorials/models/GLM4.x.md:584` |
| `ASCEND_CONNECT_TIMEOUT` | export | 2 | `docs/source/tutorials/models/DeepSeek-V4-Pro.md:167; docs/source/tutorials/models/DeepSeek-V4-Pro.md:243; docs/source/tutorials/models/MiniMax-M3.md:212; docs/source/tutorials/models/MiniMax-M3.md:255; docs/source/tutorials/models/MiniMax-M3.md:301; docs/source/tutorials/models/MiniMax-M3.md:345` |
| `ASCEND_RT_VISIBLE_DEVICES` | docker -e/--env, export, inline assignment | 25 | `docs/source/tutorials/models/DeepSeek-V3.1.md:450; docs/source/tutorials/models/DeepSeek-V3.1.md:525; docs/source/tutorials/models/DeepSeek-V3.1.md:600; docs/source/tutorials/models/DeepSeek-V3.1.md:673; docs/source/tutorials/models/DeepSeek-V3.2.md:425; docs/source/tutorials/models/DeepSeek-V3.2.md:498; docs/source/tutorials/models/DeepSeek-V3.2.md:574; docs/source/tutorials/models/DeepSeek-V3.2.md:648` |
| `ASCEND_TRANSFER_TIMEOUT` | export | 2 | `docs/source/tutorials/models/DeepSeek-V4-Pro.md:168; docs/source/tutorials/models/DeepSeek-V4-Pro.md:244; docs/source/tutorials/models/MiniMax-M3.md:213; docs/source/tutorials/models/MiniMax-M3.md:256; docs/source/tutorials/models/MiniMax-M3.md:302; docs/source/tutorials/models/MiniMax-M3.md:346` |
| `ASCEND_TRANSPORT_PRINT` | export | 4 | `docs/source/tutorials/models/DeepSeek-V3.2.md:419; docs/source/tutorials/models/DeepSeek-V3.2.md:492; docs/source/tutorials/models/DeepSeek-V3.2.md:566; docs/source/tutorials/models/DeepSeek-V3.2.md:640; docs/source/tutorials/models/GLM4.x.md:391; docs/source/tutorials/models/GLM4.x.md:454; docs/source/tutorials/models/GLM4.x.md:516; docs/source/tutorials/models/GLM4.x.md:585` |
| `CPU_AFFINITY_CONF` | export | 2 | `docs/source/tutorials/models/GLM5.2.md:873; docs/source/tutorials/models/GLM5.2.md:923; docs/source/tutorials/models/PaddleOCR-VL.md:106` |
| `HCCL_BUFFSIZE` | export, inline assignment | 24 | `docs/source/tutorials/models/DeepSeek-R1.md:244; docs/source/tutorials/models/DeepSeek-R1.md:290; docs/source/tutorials/models/DeepSeek-V3.1.md:265; docs/source/tutorials/models/DeepSeek-V3.1.md:318; docs/source/tutorials/models/DeepSeek-V3.1.md:446; docs/source/tutorials/models/DeepSeek-V3.1.md:521; docs/source/tutorials/models/DeepSeek-V3.1.md:596; docs/source/tutorials/models/DeepSeek-V3.1.md:669` |
| `HCCL_CONNECT_TIMEOUT` | export | 7 | `docs/source/tutorials/models/DeepSeek-V3.1.md:441; docs/source/tutorials/models/DeepSeek-V3.1.md:516; docs/source/tutorials/models/DeepSeek-V3.1.md:591; docs/source/tutorials/models/DeepSeek-V3.1.md:664; docs/source/tutorials/models/DeepSeek-V3.2.md:288; docs/source/tutorials/models/DeepSeek-V3.2.md:339; docs/source/tutorials/models/DeepSeek-V4-Flash.md:496; docs/source/tutorials/models/DeepSeek-V4-Flash.md:564` |
| `HCCL_EXEC_TIMEOUT` | export | 5 | `docs/source/tutorials/models/DeepSeek-V3.1.md:440; docs/source/tutorials/models/DeepSeek-V3.1.md:515; docs/source/tutorials/models/DeepSeek-V3.1.md:590; docs/source/tutorials/models/DeepSeek-V3.1.md:663; docs/source/tutorials/models/DeepSeek-V4-Flash.md:495; docs/source/tutorials/models/DeepSeek-V4-Flash.md:563; docs/source/tutorials/models/DeepSeek-V4-Flash.md:643; docs/source/tutorials/models/DeepSeek-V4-Flash.md:711` |
| `HCCL_IF_IP` | export | 16 | `docs/source/tutorials/models/DeepSeek-R1.md:139; docs/source/tutorials/models/DeepSeek-R1.md:238; docs/source/tutorials/models/DeepSeek-R1.md:284; docs/source/tutorials/models/DeepSeek-V3.1.md:150; docs/source/tutorials/models/DeepSeek-V3.1.md:259; docs/source/tutorials/models/DeepSeek-V3.1.md:312; docs/source/tutorials/models/DeepSeek-V3.1.md:433; docs/source/tutorials/models/DeepSeek-V3.1.md:508` |
| `HCCL_INTRA_PCIE_ENABLE` | export | 5 | `docs/source/tutorials/models/DeepSeek-R1.md:247; docs/source/tutorials/models/DeepSeek-R1.md:293; docs/source/tutorials/models/DeepSeek-V3.1.md:268; docs/source/tutorials/models/DeepSeek-V3.1.md:321; docs/source/tutorials/models/DeepSeek-V3.2.md:289; docs/source/tutorials/models/DeepSeek-V3.2.md:340; docs/source/tutorials/models/Kimi-K2.5.md:257; docs/source/tutorials/models/Kimi-K2.5.md:325` |
| `HCCL_INTRA_ROCE_ENABLE` | export, inline assignment | 7 | `docs/source/tutorials/models/DeepSeek-R1.md:248; docs/source/tutorials/models/DeepSeek-R1.md:294; docs/source/tutorials/models/DeepSeek-V3.1.md:269; docs/source/tutorials/models/DeepSeek-V3.1.md:322; docs/source/tutorials/models/DeepSeek-V3.2.md:290; docs/source/tutorials/models/DeepSeek-V3.2.md:341; docs/source/tutorials/models/GLM5.2.md:1084; docs/source/tutorials/models/GLM5.2.md:991` |
| `HCCL_OP_EXPANSION_MODE` | export, inline assignment | 26 | `docs/source/tutorials/models/DeepSeek-R1.md:137; docs/source/tutorials/models/DeepSeek-V3.1.md:148; docs/source/tutorials/models/DeepSeek-V3.1.md:448; docs/source/tutorials/models/DeepSeek-V3.1.md:523; docs/source/tutorials/models/DeepSeek-V3.1.md:598; docs/source/tutorials/models/DeepSeek-V3.1.md:671; docs/source/tutorials/models/DeepSeek-V3.2.md:129; docs/source/tutorials/models/DeepSeek-V3.2.md:177` |
| `HCCL_SOCKET_IFNAME` | export | 16 | `docs/source/tutorials/models/DeepSeek-R1.md:142; docs/source/tutorials/models/DeepSeek-R1.md:241; docs/source/tutorials/models/DeepSeek-R1.md:287; docs/source/tutorials/models/DeepSeek-V3.1.md:153; docs/source/tutorials/models/DeepSeek-V3.1.md:262; docs/source/tutorials/models/DeepSeek-V3.1.md:315; docs/source/tutorials/models/DeepSeek-V3.1.md:436; docs/source/tutorials/models/DeepSeek-V3.1.md:511` |
| `HCCL_TRANSFER_TIMEOUT` | export | 1 | `docs/source/tutorials/models/GLM5.2.md:1263; docs/source/tutorials/models/GLM5.2.md:1329; docs/source/tutorials/models/GLM5.2.md:136; docs/source/tutorials/models/GLM5.2.md:1395; docs/source/tutorials/models/GLM5.2.md:1468; docs/source/tutorials/models/GLM5.2.md:213; docs/source/tutorials/models/GLM5.2.md:267; docs/source/tutorials/models/GLM5.2.md:452` |
| `TASK_QUEUE_ENABLE` | export, inline assignment | 23 | `docs/source/tutorials/models/DeepSeek-V3.1.md:447; docs/source/tutorials/models/DeepSeek-V3.1.md:522; docs/source/tutorials/models/DeepSeek-V3.1.md:597; docs/source/tutorials/models/DeepSeek-V3.1.md:670; docs/source/tutorials/models/DeepSeek-V3.2.md:572; docs/source/tutorials/models/DeepSeek-V3.2.md:646; docs/source/tutorials/models/DeepSeek-V4-Flash.md:156; docs/source/tutorials/models/DeepSeek-V4-Flash.md:199` |

### 分布式通信与并行运行环境（4）

| 变量 | 使用形式 | 出现文档数 | 文档位置（示例） |
|---|---|---:|---|
| `GLOO_SOCKET_IFNAME` | export | 16 | `docs/source/tutorials/models/DeepSeek-R1.md:140; docs/source/tutorials/models/DeepSeek-R1.md:239; docs/source/tutorials/models/DeepSeek-R1.md:285; docs/source/tutorials/models/DeepSeek-V3.1.md:151; docs/source/tutorials/models/DeepSeek-V3.1.md:260; docs/source/tutorials/models/DeepSeek-V3.1.md:313; docs/source/tutorials/models/DeepSeek-V3.1.md:434; docs/source/tutorials/models/DeepSeek-V3.1.md:509` |
| `OMP_NUM_THREADS` | export, inline assignment | 24 | `docs/source/tutorials/models/DeepSeek-R1.md:243; docs/source/tutorials/models/DeepSeek-R1.md:289; docs/source/tutorials/models/DeepSeek-V3.1.md:264; docs/source/tutorials/models/DeepSeek-V3.1.md:317; docs/source/tutorials/models/DeepSeek-V3.1.md:444; docs/source/tutorials/models/DeepSeek-V3.1.md:519; docs/source/tutorials/models/DeepSeek-V3.1.md:594; docs/source/tutorials/models/DeepSeek-V3.1.md:667` |
| `OMP_PROC_BIND` | export | 23 | `docs/source/tutorials/models/DeepSeek-R1.md:242; docs/source/tutorials/models/DeepSeek-R1.md:288; docs/source/tutorials/models/DeepSeek-V3.1.md:263; docs/source/tutorials/models/DeepSeek-V3.1.md:316; docs/source/tutorials/models/DeepSeek-V3.1.md:443; docs/source/tutorials/models/DeepSeek-V3.1.md:518; docs/source/tutorials/models/DeepSeek-V3.1.md:593; docs/source/tutorials/models/DeepSeek-V3.1.md:666` |
| `TP_SOCKET_IFNAME` | export | 16 | `docs/source/tutorials/models/DeepSeek-R1.md:141; docs/source/tutorials/models/DeepSeek-R1.md:240; docs/source/tutorials/models/DeepSeek-R1.md:286; docs/source/tutorials/models/DeepSeek-V3.1.md:152; docs/source/tutorials/models/DeepSeek-V3.1.md:261; docs/source/tutorials/models/DeepSeek-V3.1.md:314; docs/source/tutorials/models/DeepSeek-V3.1.md:435; docs/source/tutorials/models/DeepSeek-V3.1.md:510` |

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
| `PYTORCH_NPU_ALLOC_CONF` | export, inline assignment | 27 | `docs/source/tutorials/models/DeepSeek-R1.md:144; docs/source/tutorials/models/DeepSeek-R1.md:245; docs/source/tutorials/models/DeepSeek-R1.md:291; docs/source/tutorials/models/DeepSeek-V3.1.md:155; docs/source/tutorials/models/DeepSeek-V3.1.md:266; docs/source/tutorials/models/DeepSeek-V3.1.md:319; docs/source/tutorials/models/DeepSeek-V3.1.md:445; docs/source/tutorials/models/DeepSeek-V3.1.md:520` |
| `TOKENIZERS_PARALLELISM` | export, inline assignment | 1 | `docs/source/tutorials/models/DeepSeekOCR2.md:128; docs/source/tutorials/models/DeepSeekOCR2.md:131` |
| `VLLM_ENGINE_READY_TIMEOUT_S` | export, inline assignment | 4 | `docs/source/tutorials/models/DeepSeek-V4-Pro.md:158; docs/source/tutorials/models/DeepSeek-V4-Pro.md:234; docs/source/tutorials/models/GLM5.2.md:874; docs/source/tutorials/models/GLM5.2.md:924; docs/source/tutorials/models/MiniMax-M3.md:210; docs/source/tutorials/models/MiniMax-M3.md:253; docs/source/tutorials/models/MiniMax-M3.md:299; docs/source/tutorials/models/MiniMax-M3.md:343` |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | export | 7 | `docs/source/tutorials/models/DeepSeek-V3.1.md:439; docs/source/tutorials/models/DeepSeek-V3.1.md:514; docs/source/tutorials/models/DeepSeek-V3.1.md:589; docs/source/tutorials/models/DeepSeek-V3.1.md:662; docs/source/tutorials/models/DeepSeek-V4-Flash.md:494; docs/source/tutorials/models/DeepSeek-V4-Flash.md:562; docs/source/tutorials/models/DeepSeek-V4-Flash.md:642; docs/source/tutorials/models/DeepSeek-V4-Flash.md:710` |
| `VLLM_HOST_IP` | export | 1 | `docs/source/tutorials/models/GLM5.2.md:1067` |
| `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` | export, inline assignment | 5 | `docs/source/tutorials/models/DeepSeek-V3.2.md:423; docs/source/tutorials/models/DeepSeek-V3.2.md:496; docs/source/tutorials/models/DeepSeek-V3.2.md:570; docs/source/tutorials/models/DeepSeek-V3.2.md:644; docs/source/tutorials/models/GLM4.x.md:520; docs/source/tutorials/models/GLM4.x.md:589; docs/source/tutorials/models/GLM5.2.md:1401; docs/source/tutorials/models/GLM5.2.md:1474` |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | export, inline assignment | 1 | `docs/source/tutorials/models/DeepSeek-V4-Flash.md:237; docs/source/tutorials/models/DeepSeek-V4-Flash.md:281; docs/source/tutorials/models/DeepSeek-V4-Flash.md:505; docs/source/tutorials/models/DeepSeek-V4-Flash.md:653` |
| `VLLM_RPC_TIMEOUT` | export, inline assignment | 7 | `docs/source/tutorials/models/DeepSeek-V3.1.md:438; docs/source/tutorials/models/DeepSeek-V3.1.md:513; docs/source/tutorials/models/DeepSeek-V3.1.md:588; docs/source/tutorials/models/DeepSeek-V3.1.md:661; docs/source/tutorials/models/DeepSeek-V4-Flash.md:493; docs/source/tutorials/models/DeepSeek-V4-Flash.md:561; docs/source/tutorials/models/DeepSeek-V4-Flash.md:641; docs/source/tutorials/models/DeepSeek-V4-Flash.md:709` |
| `VLLM_TARGET_DEVICE` | inline assignment | 2 | `docs/source/tutorials/models/Hunyuan-A13B-Instruct.md:50; docs/source/tutorials/models/Kimi-K2-Thinking.md:152` |
| `VLLM_TORCH_PROFILER_WITH_STACK` | export | 3 | `docs/source/tutorials/models/InternVL3.5.md:105; docs/source/tutorials/models/InternVL3.5.md:148; docs/source/tutorials/models/Qwen3-235B-A22B.md:410; docs/source/tutorials/models/Qwen3-235B-A22B.md:474; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:330; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:410; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:492` |
| `VLLM_USE_MODELSCOPE` | export, inline assignment | 14 | `docs/source/tutorials/models/DeepSeek-V3.2.md:831; docs/source/tutorials/models/MiniMax-M2.md:555; docs/source/tutorials/models/PaddleOCR-VL.md:103; docs/source/tutorials/models/PaddleOCR-VL.md:131; docs/source/tutorials/models/Qwen3-30B-A3B.md:257; docs/source/tutorials/models/Qwen3-Dense.md:280; docs/source/tutorials/models/Qwen3-Dense.md:302; docs/source/tutorials/models/Qwen3-Dense.md:324` |
| `VLLM_USE_V1` | export | 9 | `docs/source/tutorials/models/DeepSeek-V3.1.md:449; docs/source/tutorials/models/DeepSeek-V3.1.md:524; docs/source/tutorials/models/DeepSeek-V3.1.md:599; docs/source/tutorials/models/DeepSeek-V3.1.md:672; docs/source/tutorials/models/DeepSeek-V3.2.md:132; docs/source/tutorials/models/DeepSeek-V3.2.md:185; docs/source/tutorials/models/DeepSeek-V3.2.md:232; docs/source/tutorials/models/DeepSeek-V3.2.md:283` |
| `VLLM_WORKER_MULTIPROC_METHOD` | export | 1 | `docs/source/tutorials/models/GLM5.2.md:1267; docs/source/tutorials/models/GLM5.2.md:1333; docs/source/tutorials/models/GLM5.2.md:1399; docs/source/tutorials/models/GLM5.2.md:1472` |

### 系统与通用运行环境（3）

| 变量 | 使用形式 | 出现文档数 | 文档位置（示例） |
|---|---|---:|---|
| `LD_LIBRARY_PATH` | export, inline assignment | 10 | `docs/source/tutorials/models/DeepSeek-V3.1.md:451; docs/source/tutorials/models/DeepSeek-V3.1.md:526; docs/source/tutorials/models/DeepSeek-V3.1.md:601; docs/source/tutorials/models/DeepSeek-V3.1.md:674; docs/source/tutorials/models/GLM4.x.md:396; docs/source/tutorials/models/GLM4.x.md:459; docs/source/tutorials/models/GLM4.x.md:522; docs/source/tutorials/models/GLM4.x.md:591` |
| `LD_PRELOAD` | export, inline assignment | 9 | `docs/source/tutorials/models/DeepSeek-V4-Flash.md:154; docs/source/tutorials/models/DeepSeek-V4-Flash.md:197; docs/source/tutorials/models/DeepSeek-V4-Flash.md:233; docs/source/tutorials/models/DeepSeek-V4-Flash.md:277; docs/source/tutorials/models/DeepSeek-V4-Flash.md:503; docs/source/tutorials/models/DeepSeek-V4-Flash.md:558; docs/source/tutorials/models/DeepSeek-V4-Flash.md:651; docs/source/tutorials/models/DeepSeek-V4-Flash.md:706` |
| `PYTHONHASHSEED` | export | 2 | `docs/source/tutorials/models/GLM5.2.md:1082; docs/source/tutorials/models/GLM5.2.md:989; docs/source/tutorials/models/MiniMax-M2.md:300; docs/source/tutorials/models/MiniMax-M2.md:362` |

### 文档示例辅助变量（4）

| 变量 | 使用形式 | 出现文档数 | 文档位置（示例） |
|---|---|---:|---|
| `IMAGE` | export, inline assignment | 41 | `docs/source/tutorials/models/DeepSeek-R1.md:43; docs/source/tutorials/models/DeepSeek-R1.md:83; docs/source/tutorials/models/DeepSeek-V3.1.md:51; docs/source/tutorials/models/DeepSeek-V3.1.md:91; docs/source/tutorials/models/DeepSeek-V3.2.md:40; docs/source/tutorials/models/DeepSeek-V3.2.md:81; docs/source/tutorials/models/DeepSeek-V4-Flash.md:50; docs/source/tutorials/models/DeepSeek-V4-Flash.md:92` |
| `MODEL` | export, inline assignment | 1 | `docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md:394` |
| `MODEL_PATH` | export, inline assignment, 正文明确提及 | 8 | `docs/source/tutorials/models/Gemma4.md:82; docs/source/tutorials/models/Gemma4.md:98; docs/source/tutorials/models/Hunyuan-A13B-Instruct.md:74; docs/source/tutorials/models/Hy3-preview.md:81; docs/source/tutorials/models/LLaVA-OneVision-Qwen2-0.5B-OV.md:57; docs/source/tutorials/models/PaddleOCR-VL.md:104; docs/source/tutorials/models/PaddleOCR-VL.md:132; docs/source/tutorials/models/PaddleOCR-VL.md:94` |
| `NAME` | export, inline assignment | 10 | `docs/source/tutorials/models/DeepSeekOCR2.md:42; docs/source/tutorials/models/DeepSeekOCR2.md:76; docs/source/tutorials/models/GLM5.2.md:43; docs/source/tutorials/models/GLM5.md:49; docs/source/tutorials/models/Hy3-preview.md:35; docs/source/tutorials/models/InternVL3.5.md:34; docs/source/tutorials/models/MiniMax-M3.md:44; docs/source/tutorials/models/Mixtral-8x7B-Instruct-v0.1.md:39` |

### 模型/评测专用变量（0）

| 变量 | 使用形式 | 出现文档数 | 文档位置（示例） |
|---|---|---:|---|

### 其他文档环境变量（7）

| 变量 | 使用形式 | 出现文档数 | 文档位置（示例） |
|---|---|---:|---|
| `ACL_OP_INIT_MODE` | export, inline assignment | 5 | `docs/source/tutorials/models/DeepSeek-V3.2.md:420; docs/source/tutorials/models/DeepSeek-V3.2.md:493; docs/source/tutorials/models/DeepSeek-V3.2.md:567; docs/source/tutorials/models/DeepSeek-V3.2.md:641; docs/source/tutorials/models/DeepSeek-V4-Pro.md:157; docs/source/tutorials/models/DeepSeek-V4-Pro.md:233; docs/source/tutorials/models/GLM4.x.md:392; docs/source/tutorials/models/GLM4.x.md:455` |
| `IFNAME` | export | 2 | `docs/source/tutorials/models/DeepSeek-V4-Pro.md:147; docs/source/tutorials/models/DeepSeek-V4-Pro.md:223; docs/source/tutorials/models/MiniMax-M3.md:205; docs/source/tutorials/models/MiniMax-M3.md:248; docs/source/tutorials/models/MiniMax-M3.md:294; docs/source/tutorials/models/MiniMax-M3.md:338` |
| `IP_ADDRESS` | export | 1 | `docs/source/tutorials/models/Qwen3.5-397B-A17B.md:320; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:400; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:482` |
| `MASTER_IP_ADDRESS` | export | 1 | `docs/source/tutorials/models/Qwen3.5-397B-A17B.md:399; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:481` |
| `NETWORK_CARD_NAME` | export | 1 | `docs/source/tutorials/models/Qwen3.5-397B-A17B.md:321; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:401; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:483` |
| `NPU_MEMORY_FRACTION` | export | 1 | `docs/source/tutorials/models/gpt-oss-120b.md:104` |
| `TIKTOKEN_ENCODINGS_BASE` | export, inline assignment | 1 | `docs/source/tutorials/models/gpt-oss-120b.md:110; docs/source/tutorials/models/gpt-oss-120b.md:88` |

## 口径说明

1. 本文只覆盖 `docs/source/tutorials/models`，不扩展到源码、测试或其他文档目录。
2. 文档中的环境变量可能属于 vLLM Ascend、CANN/HCCL、PyTorch 或宿主系统；分类按主要用途组织。
3. 普通 Shell 局部赋值（如 `nic_name=...`、`MODEL_NAME=...`）不会因为被命令引用就自动算作环境变量；只有 `export`、命令前缀赋值等明确进入子进程环境的名称才纳入。
