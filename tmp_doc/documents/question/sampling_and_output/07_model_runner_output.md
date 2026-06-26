# 07. ModelRunnerOutput 是什么？

源码位置：

- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu/sample/output.py`
- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py`

本问题关注：ModelRunner 执行完 forward / sampling 后，返回给 Scheduler 的 `ModelRunnerOutput` 包含哪些字段，它和最终用户输出有什么区别。

---

## 1. 一句话回答

`ModelRunnerOutput` 是 worker 返回 Scheduler 的执行层结果，不是用户最终看到的输出。

```text
ModelRunnerOutput 解决：
  这一轮模型执行产生了什么内部结果？

RequestOutput 解决：
  用户这一轮应该看到什么文本 / token / logprobs / finished 状态？
```

---

## 2. 字段占位

后续补充 `v1/outputs.py` 中字段定义。

典型字段包括：

```text
- req_ids
- req_id_to_index
- sampled_token_ids
- logprobs
- prompt_logprobs_dict
- pooler_output
- kv_connector_output
- ec_connector_output
- num_nans_in_logits
- cudagraph_stats
- routed_experts
```

---

## 3. 为什么它不是最终输出

占位：

```text
1. sampled_token_ids 还是 token id，不是文本。
2. 它不一定包含完整 stop reason。
3. 它不负责 detokenize。
4. 它不决定 streaming delta。
5. 它还需要 Scheduler 更新 request 状态后，才能知道哪些请求 finished。
6. 它可能包含执行层专用信息，例如 KV connector、cudagraph、routed experts。
```

---

## 4. 生成路径占位

```text
GPUModelRunner.execute_model()
  → hidden states / logits / pooling output
  → sample_tokens()
  → sampled token ids / logprobs
  → maybe_get_kv_connector_output()
  → assemble ModelRunnerOutput
```

---

## 5. 跨进程特性占位

`ModelRunnerOutput` 需要从 worker / executor 返回 EngineCore，因此需要关注：

```text
- 哪些字段必须 CPU 化或可序列化；
- 哪些字段只在 driver worker 上有意义；
- pipeline parallel / tensor parallel 下字段如何聚合；
- async scheduling 下 execute_model 和 sample_tokens 如何共享状态。
```

---

## 6. 一句话总结

```text
ModelRunnerOutput 是执行层闭环的输出，不是 API 层输出；它还要经过 Scheduler 和 OutputProcessor 两次转换。
```
