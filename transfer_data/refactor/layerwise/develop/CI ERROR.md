
  =================================== FAILURES ===================================
  _ TestGVALayerTransferFailures.test_write_finish_failure_does_not_complete_layer _
  
  self = <tests.ut.distributed.ascend_store.test_kv_transfer.TestGVALayerTransferFailures testMethod=test_write_finish_failure_does_not_complete_layer>
  
      def test_write_finish_failure_does_not_complete_layer(self):
          thread, store, save_finished, task = self._make_sending_thread()
          store.batch_write_finish.return_value = [1]
      
          with self.assertRaisesRegex(RuntimeError, "batch_write_finish failed"):
  >           thread._handle_request([task])
  
  tests/ut/distributed/ascend_store/test_kv_transfer.py:270: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
      def _handle_request(  # type: ignore[override]
          self, transfer_tasks: list[LayerTransferTask]
      ):
          # Layerwise threads only run when the use_gva_layerwise gate is on,
          # i.e. the memcache backend; the assert fails fast (at thread entry)
          # if that gate invariant is ever broken, instead of surfacing as a
          # NotImplementedError from the base-class stubs on the first call.
  >       assert isinstance(self.m_store, MemcacheBackend)
  E       AssertionError
  
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:1339: AssertionError
  __ TestGVALayerTransferFailures.test_write_finish_uses_last_actual_save_task ___
  
  self = <tests.ut.distributed.ascend_store.test_kv_transfer.TestGVALayerTransferFailures testMethod=test_write_finish_uses_last_actual_save_task>
  
      def test_write_finish_uses_last_actual_save_task(self):
          thread, store, _, task = self._make_sending_thread()
          thread.final_layer_id = 1
      
  >       thread._handle_request([task])
  
  tests/ut/distributed/ascend_store/test_kv_transfer.py:279: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  self = <KVCacheStoreLayerSendingThread(KVCacheStoreLayerSendingThread, initial daemon)>
  transfer_tasks = [LayerTransferTask(layer_id=0, block_ranges=[], shared_block_data=SharedBlockData(block_ids_arr=array([0]), block_gvas...ve_keys=['k0'], load_keys=[]), group_id=0, layer_idx_in_group=0, write_finish_keys=['k0'], cached_process_tokens=None)]
  
      def _handle_request(  # type: ignore[override]
          self, transfer_tasks: list[LayerTransferTask]
      ):
          # Layerwise threads only run when the use_gva_layerwise gate is on,
          # i.e. the memcache backend; the assert fails fast (at thread entry)
          # if that gate invariant is ever broken, instead of surfacing as a
          # NotImplementedError from the base-class stubs on the first call.
  >       assert isinstance(self.m_store, MemcacheBackend)
  E       AssertionError
  
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:1339: AssertionError
  _ TestGVALayerReceivingTaskOwnership.test_empty_load_waits_for_target_layer_reuse_before_finish _
  
  self = <tests.ut.distributed.ascend_store.test_kv_transfer.TestGVALayerReceivingTaskOwnership testMethod=test_empty_load_waits_for_target_layer_reuse_before_finish>
  
      def test_empty_load_waits_for_target_layer_reuse_before_finish(self):
          load_finished_observed: list[bool] = []
          thread = None
      
          def wait_for_reuse(layer_id):
              assert thread is not None
              load_finished_observed.append(thread.layer_load_finished_events[layer_id].is_set())
      
          thread, load_finished, _, _ = self._make_thread(external_slot_release_waiter=wait_for_reuse)
          load_task = LayerLoadTask(wait_for_save_layer=None, transfer_tasks=[], layer_id=1)
          thread.request_queue.put(load_task)
      
  >       thread._handle_request(load_task)
  
  tests/ut/distributed/ascend_store/test_kv_transfer.py:418: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  self = <KVCacheStoreLayerRecvingThread(KVCacheStoreLayerRecvingThread, initial daemon)>
  data = LayerLoadTask(wait_for_save_layer=None, transfer_tasks=[], layer_id=1, attention_start_gate=None)
  
      def _handle_request(  # type: ignore[override]
          self, data: LayerLoadTask
      ):
          # Layerwise threads only run when the use_gva_layerwise gate is on,
          # i.e. the memcache backend; the assert fails fast (at thread entry)
          # if that gate invariant is ever broken, instead of surfacing as a
          # NotImplementedError from the base-class stubs on the first call.
  >       assert isinstance(self.m_store, MemcacheBackend)
  E       AssertionError
  
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:1496: AssertionError
  _ TestGVALayerReceivingTaskOwnership.test_empty_reuse_gate_waits_for_non_saving_rank_compute _
  
  self = <tests.ut.distributed.ascend_store.test_kv_transfer.TestGVALayerReceivingTaskOwnership testMethod=test_empty_reuse_gate_waits_for_non_saving_rank_compute>
  
      def test_empty_reuse_gate_waits_for_non_saving_rank_compute(self):
          thread, load_finished, save_finished, sync_events = self._make_thread()
          save_finished[0].set()
          load_task = LayerLoadTask(
              wait_for_save_layer=0,
              transfer_tasks=[],
              layer_id=1,
          )
          thread.request_queue.put(load_task)
      
  >       thread._handle_request(load_task)
  
  tests/ut/distributed/ascend_store/test_kv_transfer.py:354: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  self = <KVCacheStoreLayerRecvingThread(KVCacheStoreLayerRecvingThread, initial daemon)>
  data = LayerLoadTask(wait_for_save_layer=0, transfer_tasks=[], layer_id=1, attention_start_gate=None)
  
      def _handle_request(  # type: ignore[override]
          self, data: LayerLoadTask
      ):
          # Layerwise threads only run when the use_gva_layerwise gate is on,
          # i.e. the memcache backend; the assert fails fast (at thread entry)
          # if that gate invariant is ever broken, instead of surfacing as a
          # NotImplementedError from the base-class stubs on the first call.
  >       assert isinstance(self.m_store, MemcacheBackend)
  E       AssertionError
  
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:1496: AssertionError
  _ TestGVALayerReceivingTaskOwnership.test_h2d_waits_for_source_save_then_target_layer_reuse _
  
  self = <tests.ut.distributed.ascend_store.test_kv_transfer.TestGVALayerReceivingTaskOwnership testMethod=test_h2d_waits_for_source_save_then_target_layer_reuse>
  
      def test_h2d_waits_for_source_save_then_target_layer_reuse(self):
          call_order: list[tuple[str, int]] = []
          thread, _, save_finished, sync_events = self._make_thread(
              external_slot_release_waiter=lambda layer_id: call_order.append(("reuse", layer_id))
          )
          save_finished[0].set()
          sync_events[0].synchronize.side_effect = lambda: call_order.append(("save", 0))
      
          def record_h2d(*_args) -> int:
              call_order.append(("h2d", 1))
              return 0
      
          thread._batch_copy_with_limits = MagicMock(side_effect=record_h2d)
          task = LayerTransferTask(
              layer_id=1,
              block_ranges=[],
              shared_block_data=SharedBlockData(
                  block_ids_arr=np.asarray([0]),
                  block_gvas_arr=np.asarray([100]),
                  req_ids=["r1"],
                  is_last_chunks=[False],
              ),
          )
          load_task = LayerLoadTask(wait_for_save_layer=0, transfer_tasks=[task], layer_id=1)
          thread.request_queue.put(load_task)
      
  >       thread._handle_request(load_task)
  
  tests/ut/distributed/ascend_store/test_kv_transfer.py:402: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  self = <KVCacheStoreLayerRecvingThread(KVCacheStoreLayerRecvingThread, initial daemon)>
  data = LayerLoadTask(wait_for_save_layer=0, transfer_tasks=[LayerTransferTask(layer_id=1, block_ranges=[], shared_block_data=..._id=0, layer_idx_in_group=0, write_finish_keys=[], cached_process_tokens=None)], layer_id=1, attention_start_gate=None)
  
      def _handle_request(  # type: ignore[override]
          self, data: LayerLoadTask
      ):
          # Layerwise threads only run when the use_gva_layerwise gate is on,
          # i.e. the memcache backend; the assert fails fast (at thread entry)
          # if that gate invariant is ever broken, instead of surfacing as a
          # NotImplementedError from the base-class stubs on the first call.
  >       assert isinstance(self.m_store, MemcacheBackend)
  E       AssertionError
  
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:1496: AssertionError
  _ TestGVALayerReceivingTaskOwnership.test_handle_request_does_not_clear_worker_owned_tasks _
  
  self = <tests.ut.distributed.ascend_store.test_kv_transfer.TestGVALayerReceivingTaskOwnership testMethod=test_handle_request_does_not_clear_worker_owned_tasks>
  
      def test_handle_request_does_not_clear_worker_owned_tasks(self):
          thread, _, _, _ = self._make_thread()
          task = LayerTransferTask(
              layer_id=1,
              block_ranges=[],
              shared_block_data=SharedBlockData(
                  block_ids_arr=np.asarray([0]),
                  block_gvas_arr=np.asarray([100]),
                  req_ids=["r1"],
                  is_last_chunks=[False],
              ),
          )
          transfer_tasks = [task]
          load_task = LayerLoadTask(
              wait_for_save_layer=None,
              transfer_tasks=transfer_tasks,
              layer_id=1,
          )
          thread.request_queue.put(load_task)
      
  >       thread._handle_request(load_task)
  
  tests/ut/distributed/ascend_store/test_kv_transfer.py:340: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  self = <KVCacheStoreLayerRecvingThread(KVCacheStoreLayerRecvingThread, initial daemon)>
  data = LayerLoadTask(wait_for_save_layer=None, transfer_tasks=[LayerTransferTask(layer_id=1, block_ranges=[], shared_block_da..._id=0, layer_idx_in_group=0, write_finish_keys=[], cached_process_tokens=None)], layer_id=1, attention_start_gate=None)
  
      def _handle_request(  # type: ignore[override]
          self, data: LayerLoadTask
      ):
          # Layerwise threads only run when the use_gva_layerwise gate is on,
          # i.e. the memcache backend; the assert fails fast (at thread entry)
          # if that gate invariant is ever broken, instead of surfacing as a
          # NotImplementedError from the base-class stubs on the first call.
  >       assert isinstance(self.m_store, MemcacheBackend)
  E       AssertionError
  
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:1496: AssertionError
  _ TestGVALayerReceivingTaskOwnership.test_source_save_failure_stops_receiver_wait _
  
  self = <tests.ut.distributed.ascend_store.test_kv_transfer.TestGVALayerReceivingTaskOwnership testMethod=test_source_save_failure_stops_receiver_wait>
  
      def test_source_save_failure_stops_receiver_wait(self):
          save_failure_checker = MagicMock(side_effect=RuntimeError("save thread failed"))
          thread, _, save_finished, _ = self._make_thread(save_failure_checker=save_failure_checker)
          save_finished[0] = MagicMock()
          save_finished[0].wait.return_value = False
          load_task = LayerLoadTask(
              wait_for_save_layer=0,
              transfer_tasks=[],
              layer_id=1,
          )
      
          with self.assertRaisesRegex(RuntimeError, "save thread failed"):
  >           thread._handle_request(load_task)
  
  tests/ut/distributed/ascend_store/test_kv_transfer.py:372: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
      def _handle_request(  # type: ignore[override]
          self, data: LayerLoadTask
      ):
          # Layerwise threads only run when the use_gva_layerwise gate is on,
          # i.e. the memcache backend; the assert fails fast (at thread entry)
          # if that gate invariant is ever broken, instead of surfacing as a
          # NotImplementedError from the base-class stubs on the first call.
  >       assert isinstance(self.m_store, MemcacheBackend)
  E       AssertionError
  
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:1496: AssertionError
  =============================== warnings summary ===============================
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch/jit/_script.py:362: 14 warnings
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch/jit/_script.py:362: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
      warnings.warn(
  
  tests/ut/test_compressed_prefix_cache.py:36
    /__w/vllm-ascend/vllm-ascend/tests/ut/test_compressed_prefix_cache.py:36: PytestUnknownMarkWarning: Unknown pytest.mark.cpu_test - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
      pytestmark = pytest.mark.cpu_test
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_npu_ipc_consumer_clones_slices
    /__w/vllm-ascend/vllm-ascend/tests/ut/distributed/weight_transfer/test_packed_tensor.py:555: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      assert weights[0][1].storage().data_ptr() != weights[1][1].storage().data_ptr()
  
  -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
  =========================== short test summary info ============================
  FAILED tests/ut/distributed/ascend_store/test_kv_transfer.py::TestGVALayerTransferFailures::test_write_finish_failure_does_not_complete_layer - AssertionError
  FAILED tests/ut/distributed/ascend_store/test_kv_transfer.py::TestGVALayerTransferFailures::test_write_finish_uses_last_actual_save_task - AssertionError
  FAILED tests/ut/distributed/ascend_store/test_kv_transfer.py::TestGVALayerReceivingTaskOwnership::test_empty_load_waits_for_target_layer_reuse_before_finish - AssertionError
  FAILED tests/ut/distributed/ascend_store/test_kv_transfer.py::TestGVALayerReceivingTaskOwnership::test_empty_reuse_gate_waits_for_non_saving_rank_compute - AssertionError
  FAILED tests/ut/distributed/ascend_store/test_kv_transfer.py::TestGVALayerReceivingTaskOwnership::test_h2d_waits_for_source_save_then_target_layer_reuse - AssertionError
  FAILED tests/ut/distributed/ascend_store/test_kv_transfer.py::TestGVALayerReceivingTaskOwnership::test_handle_request_does_not_clear_worker_owned_tasks - AssertionError
  FAILED tests/ut/distributed/ascend_store/test_kv_transfer.py::TestGVALayerReceivingTaskOwnership::test_source_save_failure_stops_receiver_wait - AssertionError
  =========== 7 failed, 2826 passed, 12 skipped, 16 warnings in 56.15s ===========