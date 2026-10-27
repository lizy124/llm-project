# 02 被测代码与 PR — 阶段性归档

本任务为性能验证，未开发或修改产品代码。镜像day20261012包含vllm 0.30.0+empty、vllm-ascend 0.19.1rc2.dev2680+g080541e15.d20261011。两机安装包根目录均为/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend，已有独立git init仓库。

| 组 | 135 commit | 133 commit |
|---|---|---|
| 池化pr-18277 | 04e9b6c55a53f1d7c1eb65358a15806bd449108a | 077d7315da5daad0fd448094f2d536b7aed53412 |
| 无池化master | 67262b2d63773b200cee3ec3831a8776d21c9860 | 3811a263cbc2a53616aec6b5619c60331f69f59f |

PR #18277固定上游提交bff71237091635cb15f9b9ee1cece331eab34296。实际只应用coordinator.py、kv_transfer.py、metadata.py、pool_worker.py四个运行时文件；安装包没有tests目录，未应用PR单测。独立git init导致两机commit不同，逐轮比较全体Git跟踪文件SHA256，内容一致。完整差异、代码哈希、分支、commit保存在[evidence/20261013_pinned_revalidation_214027_retry1/01_pool_1/inputs](evidence/20261013_pinned_revalidation_214027_retry1/01_pool_1/inputs/)，镜像身份与版本在[evidence/20261013_pinned_revalidation_214027_retry1/inputs](evidence/20261013_pinned_revalidation_214027_retry1/inputs/)。

切换方法：停止目标服务后，在对应容器的安装包目录执行git checkout master或git checkout pr-18277，再重启服务。不要使用zyd_1009旧环境的rollback.py覆盖本环境。此次结束时两机均master、工作区干净，原始停止与恢复证据见05。

预热适配只读取原aisbench_test.py并在内存替换前缀子过程、跳过预热CSV保存；源代码目标块必须各精确匹配一次，否则拒绝执行。原测试脚本磁盘文件与产品代码均不修改。定向请求按原proxy协议构造P请求并把响应的kv_transfer_params交给D，正式AISBench路径保持原样。完整实现见03。
