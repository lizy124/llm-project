# 第三轮(head ee6220d7c):16 卡 e2e 超时,判定环境 flake,非本 PR

失败 job:`a3-16 card-(part 1-1)`(其余 29/30 checks 全绿)。三重证据:

1. 同一 job 在本 PR 旧 head `735065fe1` 上 success(同套件全绿)
2. `test_k3_mla_pd_tp8` 用 `MooncakeConnectorV1`,与 ascend_store 零
   import 关联;被选中是测试选择器映射,选择≠因果
3. 失败签名是基建超时而非断言;`not_ready: []` 空列表是 conftest
   `_wait_for_multiple_servers` 按 host 键控 ready 的汇报 bug(P/D 同
   host),captured log 显示 prefill(50971)600s 未就绪、进程存活

处置:PR 贴分析+更正两条评论(issuecomment-5487307147 / 5487324518);
`gh run rerun` 无 admin 权限,改为 rebase 重触发(main +18 提交,与 PR
面零重叠)→ head `2ff5cc890`。原始日志如下。

______________________________ test_k3_mla_pd_tp8 ______________________________
  
  k3_models = {'gqa': '/tmp/pytest-of-root/pytest-0/k3-dummy0/gqa', 'mla': '/tmp/pytest-of-root/pytest-0/k3-dummy0/mla', 'mla_block5': '/tmp/pytest-of-root/pytest-0/k3-dummy0/mla_block5', 'mtp': '/tmp/pytest-of-root/pytest-0/k3-dummy0/mtp', ...}
  
      def test_k3_mla_pd_tp8(k3_models: dict[str, str]) -> None:
          prefill_port, decode_port = get_open_port(), get_open_port()
          transfer_config = {
              "kv_connector": "MooncakeConnectorV1",
              "kv_connector_extra_config": {
                  "prefill": {"dp_size": 1, "tp_size": 8},
                  "decode": {"dp_size": 1, "tp_size": 8},
              },
          }
          prefill_args = _engine_args(k3_models, "mla", tp=8)
          # P also builds the draft KV that D consumes; both peers need the same
          # target/draft layer layout even though P only generates one token.
          prefill_args.pop("compilation_config")
          prefill_args["enforce_eager"] = True
          prefill_args["kv_transfer_config"] = dict(transfer_config, kv_role="kv_producer", kv_port=get_open_port())
          decode_args = _engine_args(k3_models, "mla", tp=8)
          decode_args["kv_transfer_config"] = dict(transfer_config, kv_role="kv_consumer", kv_port=get_open_port())
          servers = [
              [k3_models["target"], "--port", str(prefill_port), *_serve_args(prefill_args)],
              [k3_models["target"], "--port", str(decode_port), *_serve_args(decode_args)],
          ]
          # Use the normal P/D transfer protocol directly so missing transfer metadata
          # cannot silently fall back to local prefill and make this test pass.
  >       with RemotePDServer(servers):
  
  tests/e2e/pull_request/sixteen_card/test_kimi_k3.py:520: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  tests/e2e/conftest.py:569: in __init__
      self._wait_for_multiple_servers(
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  self = <tests.e2e.conftest.RemotePDServer object at 0xfff6a1778860>
  targets = [('127.0.0.1', 'http://127.0.0.1:50971/health'), ('127.0.0.1', 'http://127.0.0.1:41187/health')]
  timeout = 600.0, log_interval = 30.0, always_check_nodes = True
  
      def _wait_for_multiple_servers(
          self, targets, timeout: float, log_interval: float = 30.0, always_check_nodes: bool = False
      ):
          """
          targets: List[(node_ip, url)]
          log_interval
          """
          start = time.time()
          client = requests
      
          ready = {node_ip: False for node_ip, _ in targets}
      
          last_log_time = 0.0
      
          while True:
              now = time.time()
              all_ready = True
              should_log = (now - last_log_time) >= log_interval
      
              for node_ip, url in targets:
                  if ready[node_ip] and not always_check_nodes:
                      continue
      
                  try:
                      resp = client.get(url)
                      if resp.status_code == 200:
                          ready[node_ip] = True
                          logger.info("[READY] Node %s: %s is ready.", node_ip, url)
                  except RequestException:
                      all_ready = False
                      if should_log:
                          logger.debug("[WAIT] %s: connection failed", url)
      
                      # check unexpected exit
                      result = self._poll()
                      if result is not None and result != 0:
                          self._terminate_server()
                          raise RuntimeError(f"Server at {node_ip} exited unexpectedly.") from None
      
              if should_log:
                  last_log_time = now
      
              if all_ready:
                  break
      
              if now - start > timeout:
                  not_ready_nodes = [n for n, ok in ready.items() if not ok]
                  self._terminate_server()
  >               raise RuntimeError(
                      f"Timeout: these nodes did not become ready: {not_ready_nodes} in time: {timeout}s"
                  ) from None
  E               RuntimeError: Timeout: these nodes did not become ready: [] in time: 600.0s
  
  tests/e2e/conftest.py:441: RuntimeError
  ------------------------------ Captured log call -------------------------------
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  INFO     tests.e2e.conftest:conftest.py:420 [READY] Node 127.0.0.1: http://127.0.0.1:41187/health is ready.
  =============================== warnings summary ===============================
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/utils/_error_code.py:103
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/utils/_error_code.py:103: SyntaxWarning: invalid escape sequence '\['
      self.npu_exception = "\[ERROR\] [0-9\-\:]* \(PID:\d*, Device:\-?\d*, RankID:\-?\d*\) ERR\d{5}"
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/triton/backends/ascend/driver.py:332
    /usr/local/python3.12.13/lib/python3.12/site-packages/triton/backends/ascend/driver.py:332: SyntaxWarning: invalid escape sequence '\['
      for m in re.finditer('\[aicore\]', code):
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis/prof_view/prof_db_parse/_basic_db_parser.py:46
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis/prof_view/prof_db_parse/_basic_db_parser.py:46: SyntaxWarning: invalid escape sequence '\d'
      db_patten = '^msprof_\d+\.db$'
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/optim/npu_fused_rmsprop.py:16
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/optim/npu_fused_rmsprop.py:16: SyntaxWarning: invalid escape sequence '\s'
      learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where :math:`\alpha`
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:132
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:132: SyntaxWarning: invalid escape sequence '\e'
      公式1: fast_gelu(x)=$$\frac{x}{1+e^{-1.702\begin{vmatrix}x\end{vmatrix}}}e^{0.851x(x-\begin{vmatrix}x\end{vmatrix})
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:1587
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:1587: SyntaxWarning: invalid escape sequence '\_'
      softmax_layout: string类型，可选参数，用于控制TND场景下softmax的输出（softmax_max和softmax_sum）的数据排布方式。当前仅在input\_layout=“TND”时进行配置，仅支持传入“TND”。默认情况下，softmax的输出排布为NTD排布；传入TND时，softmax的输出排布为TND排布。
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:1892
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:1892: SyntaxWarning: invalid escape sequence '\_'
      softmax_layout: string类型，可选参数，用于控制TND场景下softmax的输出（softmax_max和softmax_sum）的数据排布方式。当前仅在input\_layout=“TND”时进行配置，仅支持传入“TND”。默认情况下，softmax的输出排布为NTD排布；传入TND时，softmax的输出排布为TND排布。
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:3538
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:3538: SyntaxWarning: invalid escape sequence '\e'
      $$rotate = diag(rotate1, rotate2, rotate3) = \begin{pmatrix}rotate1&0&0\\0&rotate2&0\\0&0&rotate3\\\end{pmatrix}$$
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:4156
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:4156: SyntaxWarning: invalid escape sequence '\m'
      算子功能：MhcSinkhornBackward是MhcSinkhorn的反向算子。mHC（Manifold-Constrained Hyper-Connections）架构中的MhcSinkhorn算子对输入矩阵做sinkhorn变换得到双随机矩阵$\mathbf{H}_{\text{res}}$，输出的双随机矩阵的所有元素≥0、每一行之和为1且每一列之和为1 (具有范数保持、组合封闭性和凸组合几何解释三大特性)。
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:4360
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:4360: SyntaxWarning: invalid escape sequence '\('
      outputs = swiglu\(x, dim = -1) = swish(A) * B = A * sigmoid(A) * B
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:4474
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:4474: SyntaxWarning: invalid escape sequence '\o'
      x_{l+1} = (H_{l}^{res})^{T} \times x_l + h_{l}^{out} \otimes H_{t}^{post}
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:6691
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:6691: SyntaxWarning: invalid escape sequence '\s'
      令 $B$ 表示batch size，$L_i$ 表示第i个序列的长度，$T=\sum_i^B L_i$ 表示累积序列长度。$N_k$ 表示key的头数，$N_v$ 表示value的头数，$D_k$ 表示key向量的维度，$D_v$ 表示value向量的维度。
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:6771
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:6771: SyntaxWarning: invalid escape sequence '\s'
      令 $B$ 表示batch size，$L_i$ 表示第i个序列的长度，$T=\sum_i^B L_i$ 表示累积序列长度。$N_k$ 表示key的头数，$N_v$ 表示value的头数，$D_k$ 表示key向量的维度，$D_v$ 表示value向量的维度。
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:7660
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:7660: SyntaxWarning: invalid escape sequence '\s'
      scale_value: float类型, 代表缩放系数, 用来约束梯度, 其默认值为1.0, 典型值为$\frac{1}{\sqrt{D}}$; 数据类型为float32.
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:8156
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:8156: SyntaxWarning: invalid escape sequence '\_'
      sparse模式支持sparse\_mode=4且传入mask；当sparse\_mode=4时，要求preTokens >= -actual\_seq\_qlen、nextTokens >= -actual\_seq\_kvlen、preTokens + nextTokens >= 0；
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:8691
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:8691: SyntaxWarning: invalid escape sequence '\_'
      sparse模式支持sparse\_mode=4且传入mask；当sparse\_mode=4时，要求preTokens >= -actual\_seq\_qlen、nextTokens >= -actual\_seq\_kvlen、preTokens + nextTokens >= 0；
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:10563
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:10563: SyntaxWarning: invalid escape sequence '\s'
      out(i,j)=skip1_{i,j}+skip2Optional_{i,j}+\sum_{k=0}^{K}(scales_{i,k}*(expandPermutedRows_{expandedSrcToDstRow_{i+k*num_rows},j}+bias_{expertid,j}))
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:11598
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:11598: SyntaxWarning: invalid escape sequence '\>'
      >-   H：表示嵌入向量的长度，取值\>0。
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:12042
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:12042: SyntaxWarning: invalid escape sequence '\s'
      令 $B$ 表示batch size，$L_i$ 表示第i个序列的长度，$T=\sum_i^B L_i$ 表示累积序列长度。$N_k$ 表示key的头数，$N_v$ 表示value的头数，$D_k$ 表示key向量的维度，$D_v$ 表示value向量的维度。
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:12111
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:12111: SyntaxWarning: invalid escape sequence '\['
      gamma: Device侧的Tensor类型，表示数据缩放因子；shape支持1-8维度，数据格式支持ND。shape需要满足gamma_shape = input_shape\[n:\], n < input_shape.dims()。数据类型支持FLOAT32、FLOAT16、BFLOAT16，与input数据类型保持一致。
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:12211
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:12211: SyntaxWarning: invalid escape sequence '\['
      gamma：Device侧的Tensor类型，数据缩放因子。shape支持1-8维，数据格式支持ND，数据类型支持FLOAT16、BFLOAT16。shape需要满足gamma_shape = x_shape\[n:\], n < x_shape.dims()。数据类型、数据格式需要与入参x1保持一致。不支持空tensor。
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:13392
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:13392: SyntaxWarning: invalid escape sequence '\~'
      - B（batchsize）：取值范围为1\~65536。
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:14077
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:14077: SyntaxWarning: invalid escape sequence '\_'
      v=8*ks\_max
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:14987
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:14987: SyntaxWarning: invalid escape sequence '\q'
      (1)\qquad f_t &=\sigma(W_f[h_{t-1}, x_t] + b_f) \\
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:15121
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py:15121: SyntaxWarning: invalid escape sequence '\m'
      | 输入拼接 | $\mathbf{z}_t = \begin{bmatrix} \mathbf{h}_{t-1} \\ \mathbf{x}_t \end{bmatrix}$ |
  
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch/jit/_script.py:362: 14 warnings
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch/jit/_script.py:362: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
      warnings.warn(
  
  -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
  =========================== short test summary info ============================
  FAILED tests/e2e/pull_request/sixteen_card/test_kimi_k3.py::test_k3_mla_pd_tp8 - RuntimeError: Timeout: these nodes did not become ready: [] in time: 600.0s
  ============ 1 failed, 3 passed, 39 warnings in 1238.34s (0:20:38) =============