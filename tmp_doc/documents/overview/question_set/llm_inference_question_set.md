# LLM 推理问题集

## vLLM 总览

- [x] Q001：vLLM 的核心定位是什么？它主要解决 LLM serving 中哪些问题？
  - 模块：vLLM 总览
  - 优先级：P0
  - 答案：[../answer_set/001-vllm-core-positioning.md](../answer_set/001-vllm-core-positioning.md)
  - 来源：`../llm_inference_question_driven_learning_path.md:303`
  - 参考材料：`llm-project/tmp_doc/documents/question/vllm_overview.md`，`D:/lzy/project/kv_pool/code/vllm`

## Executor / Worker / ModelRunner

- [x] Q002：Scheduler 为什么不直接管理所有 Worker，而要中间加一层 Executor？
  - 模块：Executor / Worker / ModelRunner
  - 优先级：P0
  - 答案：[../answer_set/002-scheduler-executor-boundary.md](../answer_set/002-scheduler-executor-boundary.md)
  - 来源：用户追问；`llm-project/tmp_doc/documents/question/executor_worker_model_runner/executor_worker_model_runner_overview.md:21`
  - 参考材料：`llm-project/tmp_doc/documents/question/executor_worker_model_runner/executor_worker_model_runner_overview.md`，`D:/lzy/project/kv_pool/code/vllm`
