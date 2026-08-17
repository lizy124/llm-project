# PR #13160 KV Pool Validation

## 目标

本目录用于验证 vLLM Ascend PR #13160 对 AscendStore KV Pool 基础功能的影响。

测试以已经准备好的镜像和容器为前提。容器内已经安装待测代码、运行依赖和模型权重，并且可以访问所需的 NPU、Memcache、DRAM 和 SSD 资源。本目录不负责构建镜像，主要负责保存可复现的服务启动方式、测试步骤、通过标准和执行结果。

每个测试场景都必须先证明服务按照预期配置正确启动，再执行 KV Pool 功能、精度或性能测试。仅进程存活或接口返回 HTTP 200，不能证明池化通路正确。

## 目录组织

测试首先按模型分类，再按部署拓扑和功能模式分类：

```text
pr_13160/
├── dsv4/
│   ├── pdmix_non_layerwise/
│   ├── pdmix_layerwise/
│   ├── pd_non_layerwise/
│   ├── pd_layerwise/
│   └── pd_layerwise_offload/
├── qwen3_5/
│   └── pdmix_non_layerwise/
├── glm5_2/
│   └── pdmix_non_layerwise/
├── qwen3_32b/
│   └── pdmix_non_layerwise/
└── pooling/
    └── dram_ssd/
```

目录名称中的含义：

- `pdmix`：Prefill 和 Decode 在同一个服务中运行。
- `pd`：Prefill 和 Decode 分离部署。
- `non_layerwise`：使用普通 KV Pool save/load 通路。
- `layerwise`：使用逐层 KV save/load 通路。
- `layerwise_offload`：验证 layerwise buffer reuse/offload 配置，不等同于 DRAM 到 SSD 的存储分层。
- `dram_ssd`：验证池化系统的 DRAM、SSD 写入、读取和分层存储通路。

## 场景文件

每个测试场景目录至少包含：

```text
README.md
start.sh
test.sh
```

### `start.sh`

负责正确拉起该场景需要的全部服务，包括：

- 设置模型、NPU、HCCL、vLLM Ascend、Memcache 和 Memfabric 环境变量。
- 检查模型路径、配置文件、设备数量和端口。
- 配置 `kv-transfer-config` 及相关特性开关。
- 对 PD 分离场景拉起 Prefill、Decode 和 proxy 服务。
- 等待服务就绪，并检查目标 Connector、backend 和功能模式确实生效。
- 记录最终环境、启动命令、代码版本和服务日志。
- 启动失败时返回非零退出码。

### `test.sh`

负责执行该场景的验证，包括：

- 发送基础 smoke 请求，确认模型推理正常。
- 验证首次请求的 KV 写入。
- 验证重复前缀请求的 KV 命中和加载。
- 检查输出正确性、请求状态和超时。
- 按场景检查 layerwise、PD、DRAM 或 SSD 通路。
- 检查 traceback、后台线程异常、NPU 错误和残留请求。
- 生成可判断 PASS、FAIL 或 BLOCKED 的测试结果。

### `README.md`

负责说明：

- 场景验证目标。
- 使用的镜像、容器、模型和代码 commit。
- 所需 NPU 数量及设备分配。
- 依赖的外部服务和配置文件。
- 关键环境变量和启动参数的原因。
- 执行步骤、预期现象和通过标准。
- 当前正式支持范围及已知限制。

## 执行原则

一个场景按以下顺序执行：

```text
环境和版本检查
-> 服务启动
-> readiness 检查
-> 特性生效检查
-> smoke 请求
-> KV 功能测试
-> 精度或性能测试
-> 日志和结果收集
-> 服务清理
```

服务启动和测试必须分开判断。建议使用以下结果状态：

- `PASS`：服务正确启动，且所有必需断言通过。
- `FAIL`：待测代码、功能通路、正确性或稳定性断言失败。
- `BLOCKED`：模型、容器、NPU、网络或外部存储服务异常，无法判断 PR 是否正确。
- `SKIPPED`：根据正式文档预先声明为不支持或不适用的组合。

## 能力依据

模型启动参数、功能开关和支持范围必须以 `code/vllm-ascend` 中 PR #13160 对应分支的源码和正式 `docs` 为依据，不使用个人知识库或历史测试记录作为最终能力判断。

当前正式 layerwise 文档声明 hybrid KV cache 不受支持，而 DSV4 属于 hybrid KV cache 模型。因此，DSV4 的 `pdmix_layerwise`、`pd_layerwise` 和 `pd_layerwise_offload` 场景在编写为正向测试前，需要先确认 PR #13160 是否明确新增了该能力，或者正式文档是否需要同步更新。

## 当前阶段

当前先建立测试目录和场景文件规范。后续逐个场景补充：

1. 经过实际环境验证的服务启动脚本。
2. 能够机器判定结果的测试脚本。
3. 完整的环境说明和通过标准。
4. 每次执行产生的日志、请求响应、指标和总结结果。

第一个准备完善的场景是 `dsv4/pdmix_layerwise`，但在填写实际启动命令前，需要确认使用的 DSV4 权重版本、容器环境、NPU 拓扑和该 PR 对 hybrid layerwise 的正式支持结论。
