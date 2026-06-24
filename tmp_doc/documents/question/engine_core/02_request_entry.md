# 02. 请求如何进入 EngineCore？

源码位置：`vllm/v1/engine/core.py`

本问题关注：外部请求如何进入 EngineCore，EngineCore 如何把请求交给 Scheduler，以及 abort / finish request 等控制请求如何传入内部执行流。

---

## 1. 一句话回答

TODO

```text
外部 Engine / API 将请求交给 EngineCore；
EngineCore 负责接收请求并转交给 Scheduler.add_request()，
之后请求进入 Scheduler 的 waiting / skipped_waiting 队列。
```

---

## 2. add_request 入口

TODO

---

## 3. 请求对象如何传给 Scheduler

TODO

---

## 4. abort / finish request 如何处理

TODO

---

## 5. streaming / resumable 请求入口

TODO

---

## 6. 总结

TODO
