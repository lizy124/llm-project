# 06. Structured output / grammar 如何限制采样？

源码位置：

- `vllm/vllm/sampling_params.py`
- `vllm/vllm/config/structured_outputs.py`
- `vllm/vllm/v1/request.py`
- `vllm/vllm/v1/engine/input_processor.py`
- `vllm/vllm/v1/engine/core.py`
- `vllm/vllm/v1/structured_output/__init__.py`
- `vllm/vllm/v1/structured_output/request.py`
- `vllm/vllm/v1/structured_output/backend_types.py`
- `vllm/vllm/v1/structured_output/backend_xgrammar.py`
- `vllm/vllm/v1/structured_output/utils.py`
- `vllm/vllm/v1/worker/gpu/structured_outputs.py`
- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/v1/core/sched/output.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/v1/worker/gpu/model_runner.py`
- `vllm/vllm/v1/worker/gpu/sample/sampler.py`

本问题关注：guided decoding / structured output / grammar 如何从请求参数变成 grammar FSM，Scheduler 如何在每轮采样前生成合法 token bitmask，Worker 如何用 bitmask 屏蔽 logits，采样后 Scheduler 又如何推进 grammar 状态；同时说明它和 stop condition、spec decode、reasoning、prefill chunk 的关系。

---

## 1. 一句话回答

Structured output 的核心不是“采样后检查格式”，而是“采样前约束候选 token 集合”。

主链路是：

```text
请求携带 structured_outputs 参数
  → SamplingParams 校验并选择 backend
  → Request 创建 StructuredOutputRequest
  → EngineCore.preprocess_add_request() 启动 grammar 编译
  → Scheduler 等 grammar ready 后调度请求
  → Scheduler.get_grammar_bitmask() 为本轮 scheduled requests 生成 token bitmask
  → EngineCore 把 grammar_output 传给 sample_tokens()
  → ModelRunner 在 sampler 前把非法 token logits 置为 -inf
  → sampler 在被 mask 后的 logits 上执行 temperature / top-k / top-p / sampling
  → Scheduler.update_from_output() 用实际输出 token 推进 grammar FSM
```

一句话记忆：

```text
Structured output 负责“下一步哪些 token 能选”；
Sampler 负责“在能选的 token 里怎么选”；
Stop condition 负责“选完后要不要结束”。
```

---

## 2. 本文要回答的问题

```text
1. structured_outputs 参数支持哪些约束类型？
2. backend 是什么时候选择和校验的？
3. Request 为什么会进入 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR？
4. StructuredOutputManager 负责什么？
5. grammar FSM 是如何编译、缓存和绑定到 request 的？
6. grammar bitmask 在 Scheduler 中何时生成？
7. bitmask 的维度如何和 batch / logits / spec decode 对齐？
8. Worker / ModelRunner 如何把 bitmask 应用到 logits？
9. sampler 中结构化约束和 temperature / top-k / top-p 的顺序是什么？
10. 采样后 grammar 状态在哪里推进？
11. structured output 和 stop token / EOS / stop string 的区别是什么？
12. structured output 和 speculative decoding / reasoning 有什么特殊处理？
```

---

## 3. 先给完整主链路

从一次请求进入引擎到输出 token，structured output 相关路径可以压缩成：

```text
SamplingParams( structured_outputs=... )
  → SamplingParams.verify()
  → _validate_structured_outputs()
      → 选择 backend：xgrammar / guidance / outlines / lm-format-enforcer
      → 校验 json / regex / choice / grammar / structural_tag

Request.__init__()
  → StructuredOutputRequest.from_sampling_params()
  → request.status = WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

EngineCore.preprocess_add_request()
  → structured_output_manager.grammar_init(req)
      → 首次使用时初始化 backend
      → 异步或同步 compile_grammar()
      → request.structured_output_request.grammar = Future 或 Grammar

Scheduler
  → grammar ready 后把 request 从 blocked waiting 状态提升为 WAITING
  → schedule()
  → _update_after_schedule() 标记 has_structured_output_requests
  → get_grammar_bitmask(scheduler_output)
      → structured_output_manager.grammar_bitmask(...)
      → GrammarOutput(request_ids, bitmask)

EngineCore.step()
  → execute_model(scheduler_output)
  → grammar_output = scheduler.get_grammar_bitmask(scheduler_output)
  → sample_tokens(grammar_output)

ModelRunner.sample_tokens()
  → apply_grammar_bitmask(...)
  → sampler(logits)
  → sampled token

Scheduler.update_from_output()
  → _update_request_with_output()
  → grammar.accept_tokens(req_id, new_token_ids)
  → stop / finish / output 回传
```

对应入口：

- 参数校验：`vllm/vllm/sampling_params.py:862`
- Request 创建 structured request：`vllm/vllm/v1/request.py:87`
- grammar 初始化：`vllm/vllm/v1/engine/core.py:867`
- grammar bitmask 生成：`vllm/vllm/v1/core/sched/scheduler.py:1440`
- Worker 侧应用 bitmask：`vllm/vllm/v1/worker/gpu_model_runner.py:4455`
- 采样后推进 FSM：`vllm/vllm/v1/core/sched/scheduler.py:1599`

---

## 4. structured_outputs 参数表示什么

`StructuredOutputsParams` 定义在：

```text
vllm/vllm/sampling_params.py:72
```

它支持的主要约束类型是：

```text
json              # JSON schema
regex             # 正则表达式
choice            # 只能输出某几个字符串之一
grammar           # 用户提供 grammar
json_object       # 任意 JSON object
structural_tag    # 结构化 tag 输出
```

对应字段：`vllm/vllm/sampling_params.py:72` 到 `vllm/vllm/sampling_params.py:84`

这些字段互斥。

`__post_init__()` 会校验：

```text
只能指定一种 structured output constraint；
至少要指定一种 constraint。
```

对应代码：`vllm/vllm/sampling_params.py:90` 到 `vllm/vllm/sampling_params.py:111`

所以 structured output 请求不是“多个约束叠加”，而是：

```text
每个请求选择一个结构化约束，编译成一个 grammar FSM。
```

---

## 5. structured output backend 配置

全局配置定义在：

```text
vllm/vllm/config/structured_outputs.py
```

`StructuredOutputsConfig.backend` 支持：

```text
auto
xgrammar
guidance
outlines
lm-format-enforcer
```

对应代码：`vllm/vllm/config/structured_outputs.py:12` 到 `vllm/vllm/config/structured_outputs.py:24`

重要配置包括：

```text
backend
  选择结构化输出后端。

disable_any_whitespace
  JSON 输出是否禁止任意 whitespace；只支持 xgrammar / guidance。

disable_additional_properties
  guidance backend 下是否禁用 additionalProperties。

reasoning_parser / reasoning_parser_plugin / enable_in_reasoning
  和 reasoning 模型结合时，决定结构化约束是否在 reasoning 阶段生效。
```

配置校验在：`vllm/vllm/config/structured_outputs.py:62` 到 `vllm/vllm/config/structured_outputs.py:74`

---

## 6. backend 是什么时候选择和校验的

请求参数校验入口在 `InputProcessor._validate_params()`：

```text
params.verify(
  model_config,
  speculative_config,
  structured_outputs_config,
  tokenizer,
)
```

对应代码：`vllm/vllm/v1/engine/input_processor.py:82` 到 `vllm/vllm/v1/engine/input_processor.py:100`

其中 structured output 的校验在：

```text
SamplingParams._validate_structured_outputs()
```

位置：`vllm/vllm/sampling_params.py:862`

### 6.1 基础限制

校验会先处理几类基础限制：

```text
1. diffusion LLM 暂不支持 structured outputs。
2. structured outputs 需要 tokenizer，不能 skip_tokenizer_init。
3. V1 不支持 request-level backend 混用。
4. choice 不能为空列表。
5. grammar 不能为空字符串。
```

对应代码：`vllm/vllm/sampling_params.py:871` 到 `vllm/vllm/sampling_params.py:921`

### 6.2 backend=auto 的选择策略

如果 backend 是 `auto`，校验逻辑会尝试：

```text
1. 优先 validate_xgrammar_grammar()
   成功则 backend = xgrammar

2. 如果 xgrammar 不支持：
   - 某些 Mistral tokenizer 或 guidance 不支持的 schema，fallback 到 outlines
   - 否则 fallback 到 guidance
```

对应代码：`vllm/vllm/sampling_params.py:967` 到 `vllm/vllm/sampling_params.py:1005`

这说明 backend 不是每步动态选择，而是在请求校验阶段就写入：

```text
sampling_params.structured_outputs._backend
```

---

## 7. Request 如何挂上 structured output 状态

`Request.__init__()` 中会执行：

```text
self.structured_output_request = StructuredOutputRequest.from_sampling_params(
    sampling_params
)
```

对应代码：`vllm/vllm/v1/request.py:87` 到 `vllm/vllm/v1/request.py:89`

如果请求有 structured output：

```text
request.status = WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
```

对应代码：`vllm/vllm/v1/request.py:111` 到 `vllm/vllm/v1/request.py:112`

`use_structured_output` 只是判断：

```text
self.structured_output_request is not None
```

对应代码：`vllm/vllm/v1/request.py:243`

### 7.1 StructuredOutputRequest 保存什么

`StructuredOutputRequest` 定义在：

```text
vllm/vllm/v1/structured_output/request.py:21
```

核心字段是：

```text
params
_grammar                 # Future[StructuredOutputGrammar] 或 StructuredOutputGrammar
reasoning_ended
reasoning_parser_kwargs
reasoner
```

其中 `_grammar` 可以是 Future，因为 grammar 编译可能是异步的。

### 7.2 structured_output_key

`structured_output_key` 会把请求参数标准化成：

```text
(StructuredOutputOptions, grammar_spec)
```

例如：

```text
json          → (JSON, json_str)
json_object   → (JSON_OBJECT, "")
regex         → (REGEX, regex)
choice        → (CHOICE, json.dumps(choice))
grammar       → (GRAMMAR, grammar)
structural_tag→ (STRUCTURAL_TAG, structural_tag)
```

对应代码：`vllm/vllm/v1/structured_output/request.py:77` 到 `vllm/vllm/v1/structured_output/request.py:98`

---

## 8. grammar 是如何编译的

grammar 编译由 `StructuredOutputManager` 负责。

位置：`vllm/vllm/v1/structured_output/__init__.py:36`

### 8.1 grammar_init 的触发点

`EngineCore.preprocess_add_request()` 创建 `Request` 后，如果发现请求使用 structured output：

```text
self.structured_output_manager.grammar_init(req)
```

对应代码：`vllm/vllm/v1/engine/core.py:867` 到 `vllm/vllm/v1/engine/core.py:874`

注释强调：

```text
grammar_init 只在 input processing thread 调用；
grammar 编译可以异步；
Scheduler 会在调度前检查 grammar 是否 ready。
```

### 8.2 首次使用时初始化 backend

`grammar_init()` 中，如果 `self.backend is None`，会根据请求参数里的 `_backend` 初始化一个 engine-level backend：

```text
xgrammar            → XgrammarBackend
guidance            → GuidanceBackend
outlines            → OutlinesBackend
lm-format-enforcer  → LMFormatEnforcerBackend
```

对应代码：`vllm/vllm/v1/structured_output/__init__.py:125` 到 `vllm/vllm/v1/structured_output/__init__.py:165`

这里有一个重要限制：

```text
V1 当前只支持一个 engine-level structured output backend，
不支持不同请求混用不同 backend。
```

### 8.3 异步还是同步编译

默认情况下会异步编译：

```text
self.executor.submit(self._create_grammar, request)
```

但 external launcher 模式下会同步编译，以避免 TP ranks 因 grammar ready 时间不同而死锁。

对应代码：

- 是否启用异步：`vllm/vllm/v1/structured_output/__init__.py:47` 到 `vllm/vllm/v1/structured_output/__init__.py:56`
- 编译提交：`vllm/vllm/v1/structured_output/__init__.py:167` 到 `vllm/vllm/v1/structured_output/__init__.py:171`

### 8.4 Scheduler 如何等待 grammar ready

请求初始状态是：

```text
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
```

Scheduler 在尝试提升 blocked waiting request 时，会检查：

```text
if request.status == WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
    if not (structured_output_req and structured_output_req.grammar):
        return False
    request.status = WAITING
```

对应代码：`vllm/vllm/v1/core/sched/scheduler.py:2402` 到 `vllm/vllm/v1/core/sched/scheduler.py:2407`

`StructuredOutputRequest.grammar` 属性内部会检查 Future 是否完成。

对应代码：`vllm/vllm/v1/structured_output/request.py:42` 到 `vllm/vllm/v1/structured_output/request.py:70`

所以请求不会在 grammar 未编译完成时进入正常调度。

---

## 9. StructuredOutputBackend 和 Grammar 的抽象

抽象定义在：

```text
vllm/vllm/v1/structured_output/backend_types.py
```

### 9.1 Engine-level backend

`StructuredOutputBackend` 是 engine-level 的后端对象，核心接口：

```text
compile_grammar(request_type, grammar_spec) -> StructuredOutputGrammar
allocate_token_bitmask(max_num_seqs) -> torch.Tensor
destroy()
```

对应代码：`vllm/vllm/v1/structured_output/backend_types.py:98` 到 `vllm/vllm/v1/structured_output/backend_types.py:136`

### 9.2 Request-level grammar

`StructuredOutputGrammar` 是 request-level 的状态机，核心接口：

```text
accept_tokens(request_id, tokens) -> bool
validate_tokens(tokens) -> list[int]
rollback(num_tokens)
fill_bitmask(bitmask, batch_index)
is_terminated()
reset()
```

对应代码：`vllm/vllm/v1/structured_output/backend_types.py:31` 到 `vllm/vllm/v1/structured_output/backend_types.py:95`

可以这样理解：

```text
Backend 负责编译和分配 bitmask；
Grammar 负责维护某个请求当前生成到哪一步，以及下一步哪些 token 合法。
```

---

## 10. XGrammar backend 做了什么

`XgrammarBackend` 定义在：

```text
vllm/vllm/v1/structured_output/backend_xgrammar.py:35
```

### 10.1 初始化 tokenizer info 和 compiler

初始化时会构造：

```text
xgr.TokenizerInfo
xgr.GrammarCompiler
```

对应代码：`vllm/vllm/v1/structured_output/backend_xgrammar.py:37` 到 `vllm/vllm/v1/structured_output/backend_xgrammar.py:70`

compiler 开启缓存：

```text
cache_enabled=True
cache_limit_bytes=VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024
```

### 10.2 compile_grammar

`compile_grammar()` 根据请求类型调用不同编译入口：

```text
JSON           → compile_json_schema(grammar_spec)
JSON_OBJECT    → compile_json_schema('{"type": "object"}')
GRAMMAR        → compile_grammar(grammar_spec)
REGEX          → compile_regex_with_timeout(...)
STRUCTURAL_TAG → compile_structural_tag(...)
```

对应代码：`vllm/vllm/v1/structured_output/backend_xgrammar.py:78` 到 `vllm/vllm/v1/structured_output/backend_xgrammar.py:126`

最后返回：

```text
XgrammarGrammar(
  matcher=xgr.GrammarMatcher(...),
  vocab_size=...,
  ctx=compiled_context,
)
```

### 10.3 XgrammarGrammar 如何推进状态

`XgrammarGrammar.accept_tokens()` 会逐个调用：

```text
matcher.accept_token(token)
```

成功后增加 `num_processed_tokens`，并更新：

```text
_is_terminated = matcher.is_terminated()
```

对应代码：`vllm/vllm/v1/structured_output/backend_xgrammar.py:152` 到 `vllm/vllm/v1/structured_output/backend_xgrammar.py:171`

### 10.4 validate_tokens 和 rollback

`validate_tokens()` 会临时尝试接受一串 tokens，但最后 rollback 回原状态，只返回能被接受的前缀。

对应代码：`vllm/vllm/v1/structured_output/backend_xgrammar.py:173` 到 `vllm/vllm/v1/structured_output/backend_xgrammar.py:188`

这对 spec decode 很重要：

```text
先验证 draft tokens 哪些符合 grammar，
但不能真的推进 FSM，
因为最终是否接受这些 draft token 还要等 target model 验证。
```

---

## 11. Scheduler 何时标记本轮有 structured output

Scheduler 每轮 schedule 后会调用 `_update_after_schedule()`。

位置：`vllm/vllm/v1/core/sched/scheduler.py:1130`

它会更新每个请求的 `num_computed_tokens`，并判断当前请求是否还处于 prefill chunk：

```text
request.is_prefill_chunk = request.num_computed_tokens < (
  request.num_tokens + request.num_output_placeholders
)
```

然后设置：

```text
scheduler_output.has_structured_output_requests |= (
  request.use_structured_output and not request.is_prefill_chunk
)
```

对应代码：`vllm/vllm/v1/core/sched/scheduler.py:1147` 到 `vllm/vllm/v1/core/sched/scheduler.py:1152`

这说明：

```text
prefill chunk 阶段不生成 structured output bitmask；
只有真正要采样输出 token 的阶段才需要 grammar bitmask。
```

---

## 12. Scheduler.get_grammar_bitmask() 做什么

入口：

```text
Scheduler.get_grammar_bitmask(scheduler_output)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1440`

主逻辑：

```text
1. 如果 scheduler_output.has_structured_output_requests 为 False，直接返回 None。
2. 从 scheduler_output.num_scheduled_tokens 中收集 structured output 请求 id。
3. 跳过 prefill chunk。
4. 调 structured_output_manager.grammar_bitmask(...)
5. 返回 GrammarOutput(structured_output_request_ids, bitmask)
```

对应代码：`vllm/vllm/v1/core/sched/scheduler.py:1440` 到 `vllm/vllm/v1/core/sched/scheduler.py:1462`

`GrammarOutput` 里保存两样东西：

```text
structured_output_request_ids
grammar_bitmask
```

它跟 `SchedulerOutput` 分开，是因为：

```text
SchedulerOutput 先给 Worker 启动 forward；
grammar bitmask 可以在 forward 期间并行计算；
最后 sample_tokens() 再消费 grammar_output。
```

`EngineCore.step()` 的顺序也体现了这个并行意图：

```text
future = model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = scheduler.get_grammar_bitmask(scheduler_output)
model_output = future.result()
if model_output is None:
    model_output = model_executor.sample_tokens(grammar_output)
```

对应代码：`vllm/vllm/v1/engine/core.py:490` 到 `vllm/vllm/v1/engine/core.py:500`

---

## 13. grammar_bitmask 的 shape 和含义

`StructuredOutputManager.grammar_bitmask()` 定义在：

```text
vllm/vllm/v1/structured_output/__init__.py:204
```

### 13.1 bitmask 分配

第一次使用时，它会分配：

```text
max_batch_size * (1 + max_num_spec_tokens)
```

行 bitmask。

对应代码：`vllm/vllm/v1/structured_output/__init__.py:214` 到 `vllm/vllm/v1/structured_output/__init__.py:226`

含义是：

```text
非 spec decode：每个 request 需要 1 行 bitmask。

spec decode：每个 request 可能需要：
  num_spec_tokens 行 draft token 的 bitmask
  + 1 行 bonus / normal sampled token 的 bitmask
```

### 13.2 每一行 bitmask 表示什么

每一行是 packed int32 bitmask：

```text
shape roughly = [num_masks, ceil(vocab_size / 32)]
```

每个 bit 表示一个 token 是否允许。

在 worker 侧应用时，非法 token logits 会被写成：

```text
-inf
```

### 13.3 spec decode 下如何临时推进 FSM

对于每个 structured output request，manager 会遍历：

```text
scheduled_spec_decode_tokens[req_id] + [-1]
```

每一步：

```text
1. 先 fill 当前 grammar 状态下的 bitmask。
2. 如果 token 不是 -1，则临时 accept_tokens(token)，推进 FSM。
3. 累计推进次数。
4. 处理完后 rollback(state_advancements)。
```

对应代码：`vllm/vllm/v1/structured_output/__init__.py:275` 到 `vllm/vllm/v1/structured_output/__init__.py:295`

这个设计解决了 spec decode 的问题：

```text
要为 draft token 的每个位置生成合法 token mask，
就必须模拟 FSM 前进；
但这些 draft token 尚未被最终接受，
所以 bitmask 生成完必须 rollback。
```

---

## 14. reasoning 场景下什么时候启用 bitmask

Structured output 可以和 reasoning parser 结合。

`StructuredOutputManager.should_fill_bitmask()` 决定本轮是否真的填 grammar bitmask。

位置：`vllm/vllm/v1/structured_output/__init__.py:305`

规则是：

```text
如果没有 reasoner：
  直接填 bitmask。

如果 enable_in_reasoning=True：
  reasoning 阶段也填 bitmask。

否则：
  只有 reasoning 已结束后才填 bitmask。
```

对应代码：`vllm/vllm/v1/structured_output/__init__.py:305` 到 `vllm/vllm/v1/structured_output/__init__.py:323`

这解决的是“思考内容”和“最终结构化答案”之间的边界问题：

```text
有些模型先输出 reasoning，再输出 JSON；
如果过早施加 JSON grammar，会把 reasoning 阶段也限制成 JSON。
```

---

## 15. Worker 如何把 grammar bitmask 应用到 logits

vLLM 里有两条相关路径：

```text
GPUModelRunner 路径：
  vllm/vllm/v1/structured_output/utils.py::apply_grammar_bitmask

V2 GPU runner 路径：
  vllm/vllm/v1/worker/gpu/structured_outputs.py::StructuredOutputsWorker
```

两者目标相同：

```text
在 sampler 运行之前，把非法 token 的 logits 置为 -inf。
```

---

## 16. GPUModelRunner 路径：apply_grammar_bitmask

`GPUModelRunner.sample_tokens()` 中，在 `_sample()` 之前：

```text
if grammar_output is not None:
    apply_grammar_bitmask(
        scheduler_output,
        grammar_output,
        self.input_batch,
        logits,
    )
```

对应代码：`vllm/vllm/v1/worker/gpu_model_runner.py:4455` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4459`

### 16.1 为什么要重新排序 bitmask

Scheduler 生成的 bitmask 顺序是：

```text
grammar_output.structured_output_request_ids
```

但 GPU worker 的 batch 顺序是：

```text
input_batch.req_ids
```

两者不一定一致。

所以 `apply_grammar_bitmask()` 会先构造：

```text
struct_out_req_batch_indices
out_indices
sorted_bitmask
```

对应代码：`vllm/vllm/v1/structured_output/utils.py:103` 到 `vllm/vllm/v1/structured_output/utils.py:140`

### 16.2 spec decode 下 logits index 如何对齐

`apply_grammar_bitmask()` 会考虑：

```text
scheduler_output.scheduled_spec_decode_tokens
```

每个 request 的 logits 位置会因为 draft tokens 增加 offset：

```text
logit_index = batch_index + cumulative_offset
cumulative_offset += len(spec_tokens.get(req_id, ()))
```

对应代码：`vllm/vllm/v1/structured_output/utils.py:112` 到 `vllm/vllm/v1/structured_output/utils.py:120`

然后对每个 structured request 写入：

```text
1 + num_spec_tokens
```

行 bitmask。

对应代码：`vllm/vllm/v1/structured_output/utils.py:132` 到 `vllm/vllm/v1/structured_output/utils.py:140`

### 16.3 最终如何 mask logits

GPU 情况下调用：

```text
xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)
```

对应代码：`vllm/vllm/v1/structured_output/utils.py:150` 到 `vllm/vllm/v1/structured_output/utils.py:161`

CPU 情况也会调用 xgrammar 的 CPU mask 逻辑，只是必要时把 logits 转成 float32 后再拷回。

对应代码：`vllm/vllm/v1/structured_output/utils.py:163` 到 `vllm/vllm/v1/structured_output/utils.py:174`

---

## 17. V2 GPU runner 路径：StructuredOutputsWorker

`StructuredOutputsWorker` 定义在：

```text
vllm/vllm/v1/worker/gpu/structured_outputs.py:12
```

初始化时预分配：

```text
logits_indices: [max_num_logits]
grammar_bitmask: [max_num_logits, ceil(vocab_size / 32)]
copy_stream
```

对应代码：`vllm/vllm/v1/worker/gpu/structured_outputs.py:12` 到 `vllm/vllm/v1/worker/gpu/structured_outputs.py:21`

V2 GPU runner 在 sample 中调用：

```text
self.structured_outputs_worker.apply_grammar_bitmask(
  logits,
  input_batch,
  grammar_output.structured_output_request_ids,
  grammar_output.grammar_bitmask,
)
```

对应代码：`vllm/vllm/v1/worker/gpu/model_runner.py:1037` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:1053`

### 17.1 mapping 如何构造

它会根据：

```text
input_batch.req_ids
input_batch.cu_num_logits_np
```

把每个 structured request 映射到 logits 行区间：

```text
logits_start_idx = cu_num_logits[req_idx]
logits_end_idx = cu_num_logits[req_idx + 1]
mapping.extend(range(logits_start_idx, logits_end_idx))
```

对应代码：`vllm/vllm/v1/worker/gpu/structured_outputs.py:39` 到 `vllm/vllm/v1/worker/gpu/structured_outputs.py:48`

### 17.2 Triton kernel 如何 mask

`_apply_grammar_bitmask_kernel` 会：

```text
1. 读取 packed int32 bitmask。
2. unpack 成 BLOCK_SIZE 个 bool。
3. 对非法 token 位置写入 -inf。
```

对应代码：`vllm/vllm/v1/worker/gpu/structured_outputs.py:85` 到 `vllm/vllm/v1/worker/gpu/structured_outputs.py:115`

---

## 18. sampler 中 structured output 的位置

structured output mask 发生在 sampler 之前。

之后 `Sampler.__call__()` 会进入正常采样流程：

```text
Sampler.__call__()
  → sample()
  → apply_sampling_params()
      → logit_bias
      → penalties
      → bad_words
      → temperature
      → min_p
  → top_k / top_p
  → gumbel_sample 或 flashinfer_sample
```

对应代码：

- sampler 入口：`vllm/vllm/v1/worker/gpu/sample/sampler.py:72`
- sampling params 应用：`vllm/vllm/v1/worker/gpu/sample/sampler.py:146`
- sample 主流程：`vllm/vllm/v1/worker/gpu/sample/sampler.py:198`

因此结构化约束和其他采样参数的关系是：

```text
先由 grammar bitmask 删除非法 token；
再在剩余 token 上应用普通采样策略。
```

如果某个 token 被 grammar 禁止，即使它原始 logits 很高，也会变成 `-inf`，不会进入后续 top-k / top-p / sampling。

---

## 19. 采样后 grammar 状态在哪里推进

采样完成后，`Scheduler.update_from_output()` 会更新 request。

位置：`vllm/vllm/v1/core/sched/scheduler.py:1464`

核心顺序是：

```text
1. 从 ModelRunnerOutput 取 generated_token_ids。
2. 调 _update_request_with_output() 写入 request 输出，并检查 stop。
3. 如果请求需要 structured output advance：
   grammar.accept_tokens(req_id, new_token_ids)
4. 如果 grammar 拒绝，则 request FINISHED_ERROR。
```

对应代码：`vllm/vllm/v1/core/sched/scheduler.py:1589` 到 `vllm/vllm/v1/core/sched/scheduler.py:1615`

这说明：

```text
grammar 状态推进发生在 Scheduler 回收真实输出后，
不是 Worker sampler 内部。
```

这样做的好处是：

```text
Scheduler 持有 request 的权威状态；
spec decode 的接受 / 拒绝、stop、preemption 等都能统一对账。
```

---

## 20. speculative decoding 下的特殊处理

Structured output 和 spec decode 有两个关键交互。

### 20.1 draft token 需要先 validate

当 worker 生成 draft tokens 后，Scheduler 会调用：

```text
update_draft_token_ids()
```

如果请求使用 structured output，就执行：

```text
spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
```

对应代码：`vllm/vllm/v1/core/sched/scheduler.py:1896` 到 `vllm/vllm/v1/core/sched/scheduler.py:1916`

这会过滤掉不符合 grammar 的 draft token。

### 20.2 已经进入 SchedulerOutput 的 spec tokens 也要过滤

batch queue / async 场景下，可能需要在已生成的 `SchedulerOutput` 里更新 draft tokens。

`update_draft_token_ids_in_output()` 会：

```text
1. 截断 draft tokens 到本轮 scheduled 数量。
2. validate_tokens() 过滤不符合 grammar 的前缀。
3. 对无效位置补 -1。
4. 记录 num_invalid_spec_tokens。
```

对应代码：`vllm/vllm/v1/core/sched/scheduler.py:1918` 到 `vllm/vllm/v1/core/sched/scheduler.py:1954`

EngineCore 的 batch queue 路径也说明：在计算 deferred request 的 grammar bitmask 前，要先过滤 draft tokens。

对应代码：`vllm/vllm/v1/engine/core.py:612` 到 `vllm/vllm/v1/engine/core.py:629`

### 20.3 bitmask 生成时模拟 draft token 推进

如前文所述，`grammar_bitmask()` 会为 spec tokens 临时 accept，再 rollback。

对应代码：`vllm/vllm/v1/structured_output/__init__.py:275` 到 `vllm/vllm/v1/structured_output/__init__.py:295`

所以 spec decode 下的原则是：

```text
validate_tokens 用于过滤 draft token；
grammar_bitmask 用于约束 target / bonus 位置；
accept_tokens 只在最终输出回收时真正推进 FSM。
```

---

## 21. structured output 和 stop condition 的区别

Structured output 和 stop condition 经常容易混淆。

### 21.1 structured output

它回答的是：

```text
当前 grammar 状态下，下一步哪些 token 可以被采样？
```

作用点：

```text
采样前 logits mask。
```

### 21.2 stop condition

它回答的是：

```text
已经生成了这些 token，请求是否应该结束？
```

作用点：

```text
采样后 Scheduler.update_from_output()。
```

### 21.3 二者可以同时存在

例如 JSON schema 约束会限制每一步 token 必须形成合法 JSON；但请求仍然可能因为：

```text
EOS
stop token
stop string
max_tokens
length cap
```

结束。

所以：

```text
structured output 不替代 stop；
stop 也不替代 structured output。
```

---

## 22. 为什么 prefill chunk 不生成 grammar bitmask

Scheduler 标记 structured output 请求时有条件：

```text
request.use_structured_output and not request.is_prefill_chunk
```

对应代码：`vllm/vllm/v1/core/sched/scheduler.py:1150` 到 `vllm/vllm/v1/core/sched/scheduler.py:1152`

原因是：

```text
prefill 阶段是在处理 prompt，不是在采样新的输出 token；
grammar bitmask 只用于限制“下一步采样候选 token”。
```

如果 chunked prefill 还没到最后一段，不会产生需要返回给用户的新 token，也不需要 grammar mask。

---

## 23. bitmask 为空、终止和 full mask

`StructuredOutputManager._fill_bitmasks()` 中有一个分支：

```text
if apply_bitmask and not grammar.is_terminated():
    grammar.fill_bitmask(...)
else:
    bitmask[index].fill_(self._full_mask)
```

对应代码：`vllm/vllm/v1/structured_output/__init__.py:186` 到 `vllm/vllm/v1/structured_output/__init__.py:197`

`_full_mask` 是：

```text
torch.tensor(-1, dtype=torch.int32)
```

对应代码：`vllm/vllm/v1/structured_output/__init__.py:58` 到 `vllm/vllm/v1/structured_output/__init__.py:59`

这表示：

```text
当不需要应用 grammar 或 grammar 已 terminated 时，
该行 bitmask 允许所有 token，
不再额外限制 logits。
```

如果最终真实输出 token 被 grammar 拒绝，Scheduler 会把请求置为：

```text
FINISHED_ERROR
```

对应代码：`vllm/vllm/v1/core/sched/scheduler.py:1603` 到 `vllm/vllm/v1/core/sched/scheduler.py:1615`

---

## 24. 几个容易混淆的点

### 24.1 grammar bitmask 不负责 detokenize

bitmask 工作在 token id 维度：

```text
[vocab token id] -> allowed / disallowed
```

它不负责把 token 还原成字符串，也不负责做 stop string 匹配。

### 24.2 grammar 编译不在每一步重复做

grammar 编译在请求进入时由 `grammar_init()` 启动。

每一步只是：

```text
根据当前 FSM 状态 fill_bitmask。
```

### 24.3 Scheduler 持有 grammar 权威状态

Worker 只是使用 bitmask mask logits。

真正的 FSM 推进发生在 Scheduler 回收输出后。

### 24.4 request id 顺序和 GPU batch 顺序可能不同

所以 worker 侧必须把 Scheduler 生成的 bitmask 重新对齐到 `input_batch.req_ids` / logits 行。

### 24.5 spec decode 下不能直接 advance grammar

draft token 还不一定被接受，因此只能：

```text
validate_tokens 或临时 accept + rollback。
```

最终 accepted tokens 才会在 `update_from_output()` 中推进 grammar。

### 24.6 structured output 不等于 allowed_token_ids

`allowed_token_ids` 更像静态 logit bias / token 白名单。

structured output 是动态 FSM：

```text
合法 token 集合会随着已生成 token 不断变化。
```

---

## 25. 最终可以记成一张表

| 阶段 | 关键代码 | 核心产物 | 作用 |
|---|---|---|---|
| 请求参数 | `StructuredOutputsParams` | json / regex / choice / grammar 等 | 表达用户想要的结构化约束 |
| 参数校验 | `_validate_structured_outputs()` | `_backend` | 校验约束合法性并选择 backend |
| 请求创建 | `Request.__init__()` | `StructuredOutputRequest` | 把 structured output 状态挂到 request 上 |
| grammar 初始化 | `StructuredOutputManager.grammar_init()` | Future 或 Grammar | 异步 / 同步编译 request-level grammar |
| 等待 ready | `_try_promote_blocked_waiting_request()` | WAITING 状态 | grammar ready 后请求才可调度 |
| 调度标记 | `_update_after_schedule()` | `has_structured_output_requests` | 标记本轮是否需要 grammar bitmask |
| bitmask 生成 | `get_grammar_bitmask()` / `grammar_bitmask()` | `GrammarOutput` | 为本轮采样位置生成合法 token mask |
| Worker 对齐 | `apply_grammar_bitmask()` | sorted bitmask / logits indices | 将 request 顺序对齐到 GPU batch / logits 行 |
| logits mask | xgrammar / Triton kernel | masked logits | 非法 token logits 变成 `-inf` |
| 采样 | `Sampler.__call__()` | sampled token | 在被约束后的 logits 上执行采样 |
| 状态推进 | `grammar.accept_tokens()` | 更新 FSM | 用真实输出 token 推进 grammar 状态 |
| spec 过滤 | `validate_tokens()` | 合法 draft 前缀 | 过滤不符合 grammar 的 draft tokens |

---

## 26. 一句话总结

Structured output 在 vLLM V1 中是一条“请求级 grammar FSM + 每步 bitmask + 采样前 logits mask + 采样后 FSM 推进”的链路：

```text
请求参数决定 grammar；
EngineCore 负责启动 grammar 编译；
Scheduler 负责等待 grammar ready、生成 bitmask、推进 FSM；
Worker / ModelRunner 负责把 bitmask 应用到 logits；
Sampler 只在被允许的 token 集合中采样。
```

如果只记住一句话，就是：

```text
structured output 是采样前的动态 token 合法性约束；它约束候选集合，不负责停止判断，也不在 Worker 中维护最终 grammar 状态。
```
