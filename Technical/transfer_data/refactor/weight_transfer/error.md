 
  =================================== FAILURES ===================================
  _________________ test_packed_broadcast_producer_single_tensor _________________
  
      def test_packed_broadcast_producer_single_tensor():
          """A single small tensor is broadcast exactly once with correct bytes."""
          tensor = torch.arange(12, dtype=torch.float32)
          iterator = iter([("w", tensor)])
          group = _make_group_mock()
      
  >       packed_broadcast_producer(
              iterator=iterator,
              group=group,
              src=0,
              post_iter_func=lambda item: item[1],
          )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:107: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f78507f0d00>
  group = <MagicMock id='140154723314496'>, src = 0
  post_iter_func = <function test_packed_broadcast_producer_single_tensor.<locals>.<lambda> at 0x7f785071c680>
  buffer_size_bytes = 1073741824, num_buffers = 2
  
      def packed_broadcast_producer(
          iterator: Iterator[tuple[str, torch.Tensor]],
          group: Any,
          src: int,
          post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
          num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
      ) -> None:
          """Broadcast tensors in a packed manner from trainer to workers.
      
          Args:
              iterator: Iterator of model parameters. Returns a tuple of (name, tensor)
              group: Process group (PyHcclCommunicator)
              src: Source rank (0 in current implementation)
              post_iter_func: Function to apply to each (name, tensor) pair before
                             packing, should return a tensor
              buffer_size_bytes: Size in bytes for each packed tensor buffer.
                                Both producer and consumer must use the same value.
              num_buffers: Number of buffers for double/triple buffering.
                          Both producer and consumer must use the same value.
          """
          target_packed_tensor_size = buffer_size_bytes
      
          streams = [torch.npu.Stream() for _ in range(num_buffers)]
          buffer_idx = 0
      
          packing_tensor_list: list[list[torch.Tensor]] = [[] for _ in range(num_buffers)]
          packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
  >       packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:46: ModuleNotFoundError
  __________ test_packed_broadcast_producer_multiple_tensors_one_buffer __________
  
      def test_packed_broadcast_producer_multiple_tensors_one_buffer():
          """Multiple tensors under the buffer size are packed into one broadcast."""
          tensors = [
              ("a", torch.full((4,), 1.0, dtype=torch.float32)),
              ("b", torch.full((8,), 2.0, dtype=torch.float32)),
          ]
          group = _make_group_mock()
      
  >       packed_broadcast_producer(
              iterator=iter(tensors),
              group=group,
              src=0,
              post_iter_func=lambda item: item[1],
          )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:131: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f78507e8400>
  group = <MagicMock id='140154723319536'>, src = 0
  post_iter_func = <function test_packed_broadcast_producer_multiple_tensors_one_buffer.<locals>.<lambda> at 0x7f785071cfe0>
  buffer_size_bytes = 1073741824, num_buffers = 2
  
      def packed_broadcast_producer(
          iterator: Iterator[tuple[str, torch.Tensor]],
          group: Any,
          src: int,
          post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
          num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
      ) -> None:
          """Broadcast tensors in a packed manner from trainer to workers.
      
          Args:
              iterator: Iterator of model parameters. Returns a tuple of (name, tensor)
              group: Process group (PyHcclCommunicator)
              src: Source rank (0 in current implementation)
              post_iter_func: Function to apply to each (name, tensor) pair before
                             packing, should return a tensor
              buffer_size_bytes: Size in bytes for each packed tensor buffer.
                                Both producer and consumer must use the same value.
              num_buffers: Number of buffers for double/triple buffering.
                          Both producer and consumer must use the same value.
          """
          target_packed_tensor_size = buffer_size_bytes
      
          streams = [torch.npu.Stream() for _ in range(num_buffers)]
          buffer_idx = 0
      
          packing_tensor_list: list[list[torch.Tensor]] = [[] for _ in range(num_buffers)]
          packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
  >       packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:46: ModuleNotFoundError
  _________ test_packed_broadcast_producer_splits_when_exceeding_buffer __________
  
      def test_packed_broadcast_producer_splits_when_exceeding_buffer():
          """Tensors larger than ``buffer_size_bytes`` trigger an extra broadcast."""
          # buffer = 32 bytes (8 float32). Two tensors of 12 floats each (48 bytes)
          # → first tensor fills past the threshold and triggers a broadcast after it.
          tensors = [
              ("a", torch.full((12,), 1.0, dtype=torch.float32)),
              ("b", torch.full((12,), 2.0, dtype=torch.float32)),
          ]
          group = _make_group_mock()
      
  >       packed_broadcast_producer(
              iterator=iter(tensors),
              group=group,
              src=0,
              post_iter_func=lambda item: item[1],
              buffer_size_bytes=32,
          )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:157: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f78507f27a0>
  group = <MagicMock id='140154723315360'>, src = 0
  post_iter_func = <function test_packed_broadcast_producer_splits_when_exceeding_buffer.<locals>.<lambda> at 0x7f785071ce00>
  buffer_size_bytes = 32, num_buffers = 2
  
      def packed_broadcast_producer(
          iterator: Iterator[tuple[str, torch.Tensor]],
          group: Any,
          src: int,
          post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
          num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
      ) -> None:
          """Broadcast tensors in a packed manner from trainer to workers.
      
          Args:
              iterator: Iterator of model parameters. Returns a tuple of (name, tensor)
              group: Process group (PyHcclCommunicator)
              src: Source rank (0 in current implementation)
              post_iter_func: Function to apply to each (name, tensor) pair before
                             packing, should return a tensor
              buffer_size_bytes: Size in bytes for each packed tensor buffer.
                                Both producer and consumer must use the same value.
              num_buffers: Number of buffers for double/triple buffering.
                          Both producer and consumer must use the same value.
          """
          target_packed_tensor_size = buffer_size_bytes
      
          streams = [torch.npu.Stream() for _ in range(num_buffers)]
          buffer_idx = 0
      
          packing_tensor_list: list[list[torch.Tensor]] = [[] for _ in range(num_buffers)]
          packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
  >       packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:46: ModuleNotFoundError
  ________________ test_packed_broadcast_producer_empty_iterator _________________
  
      def test_packed_broadcast_producer_empty_iterator():
          """An empty iterator triggers no broadcasts and does not hang."""
          group = _make_group_mock()
  >       packed_broadcast_producer(
              iterator=iter([]),
              group=group,
              src=0,
              post_iter_func=lambda item: item[1],
          )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:178: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f7850803280>
  group = <MagicMock id='140154715189472'>, src = 0
  post_iter_func = <function test_packed_broadcast_producer_empty_iterator.<locals>.<lambda> at 0x7f78506d5a80>
  buffer_size_bytes = 1073741824, num_buffers = 2
  
      def packed_broadcast_producer(
          iterator: Iterator[tuple[str, torch.Tensor]],
          group: Any,
          src: int,
          post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
          num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
      ) -> None:
          """Broadcast tensors in a packed manner from trainer to workers.
      
          Args:
              iterator: Iterator of model parameters. Returns a tuple of (name, tensor)
              group: Process group (PyHcclCommunicator)
              src: Source rank (0 in current implementation)
              post_iter_func: Function to apply to each (name, tensor) pair before
                             packing, should return a tensor
              buffer_size_bytes: Size in bytes for each packed tensor buffer.
                                Both producer and consumer must use the same value.
              num_buffers: Number of buffers for double/triple buffering.
                          Both producer and consumer must use the same value.
          """
          target_packed_tensor_size = buffer_size_bytes
      
          streams = [torch.npu.Stream() for _ in range(num_buffers)]
          buffer_idx = 0
      
          packing_tensor_list: list[list[torch.Tensor]] = [[] for _ in range(num_buffers)]
          packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
  >       packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:46: ModuleNotFoundError
  __________ test_packed_broadcast_producer_uses_num_buffers_to_rotate ___________
  
      def test_packed_broadcast_producer_uses_num_buffers_to_rotate():
          """The producer cycles through ``num_buffers`` streams."""
          tensors = [(f"w{i}", torch.zeros(1, dtype=torch.float32)) for i in range(6)]
          group = _make_group_mock()
          # Small buffer so each tensor triggers its own broadcast.
  >       packed_broadcast_producer(
              iterator=iter(tensors),
              group=group,
              src=0,
              post_iter_func=lambda item: item[1],
              buffer_size_bytes=1,  # 1 byte → every 4-byte tensor overflows immediately
              num_buffers=2,
          )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:192: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f785002a740>
  group = <MagicMock id='140154715154304'>, src = 0
  post_iter_func = <function test_packed_broadcast_producer_uses_num_buffers_to_rotate.<locals>.<lambda> at 0x7f78506d5d00>
  buffer_size_bytes = 1, num_buffers = 2
  
      def packed_broadcast_producer(
          iterator: Iterator[tuple[str, torch.Tensor]],
          group: Any,
          src: int,
          post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
          num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
      ) -> None:
          """Broadcast tensors in a packed manner from trainer to workers.
      
          Args:
              iterator: Iterator of model parameters. Returns a tuple of (name, tensor)
              group: Process group (PyHcclCommunicator)
              src: Source rank (0 in current implementation)
              post_iter_func: Function to apply to each (name, tensor) pair before
                             packing, should return a tensor
              buffer_size_bytes: Size in bytes for each packed tensor buffer.
                                Both producer and consumer must use the same value.
              num_buffers: Number of buffers for double/triple buffering.
                          Both producer and consumer must use the same value.
          """
          target_packed_tensor_size = buffer_size_bytes
      
          streams = [torch.npu.Stream() for _ in range(num_buffers)]
          buffer_idx = 0
      
          packing_tensor_list: list[list[torch.Tensor]] = [[] for _ in range(num_buffers)]
          packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
  >       packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:46: ModuleNotFoundError
  ________________ test_packed_broadcast_producer_passes_src_rank ________________
  
      def test_packed_broadcast_producer_passes_src_rank():
          """``src`` is forwarded to ``group.broadcast`` unchanged."""
          group = _make_group_mock()
  >       packed_broadcast_producer(
              iterator=iter([("w", torch.zeros(4, dtype=torch.float32))]),
              group=group,
              src=3,
              post_iter_func=lambda item: item[1],
              buffer_size_bytes=1,
          )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:207: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f7850812680>
  group = <MagicMock id='140154715156272'>, src = 3
  post_iter_func = <function test_packed_broadcast_producer_passes_src_rank.<locals>.<lambda> at 0x7f78506d5e40>
  buffer_size_bytes = 1, num_buffers = 2
  
      def packed_broadcast_producer(
          iterator: Iterator[tuple[str, torch.Tensor]],
          group: Any,
          src: int,
          post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
          num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
      ) -> None:
          """Broadcast tensors in a packed manner from trainer to workers.
      
          Args:
              iterator: Iterator of model parameters. Returns a tuple of (name, tensor)
              group: Process group (PyHcclCommunicator)
              src: Source rank (0 in current implementation)
              post_iter_func: Function to apply to each (name, tensor) pair before
                             packing, should return a tensor
              buffer_size_bytes: Size in bytes for each packed tensor buffer.
                                Both producer and consumer must use the same value.
              num_buffers: Number of buffers for double/triple buffering.
                          Both producer and consumer must use the same value.
          """
          target_packed_tensor_size = buffer_size_bytes
      
          streams = [torch.npu.Stream() for _ in range(num_buffers)]
          buffer_idx = 0
      
          packing_tensor_list: list[list[torch.Tensor]] = [[] for _ in range(num_buffers)]
          packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
  >       packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:46: ModuleNotFoundError
  _________________ test_packed_broadcast_consumer_single_tensor _________________
  
      def test_packed_broadcast_consumer_single_tensor():
          """Consumer unpacks a single broadcasted tensor and loads it."""
          # Arrange: producer-style payload (12 float32 = 48 bytes, uint8 view)
          original = torch.arange(12, dtype=torch.float32)
          packed = original.view(torch.uint8).view(-1).clone()
          # Group returns ``packed`` on broadcast.
          group = _make_group_mock()
          group.broadcast = MagicMock(side_effect=lambda t, **kw: t.copy_(packed))
      
          received: list[tuple[str, torch.Tensor]] = []
      
  >       packed_broadcast_consumer(
              iterator=iter([("w", ([12], torch.float32))]),
              group=group,
              src=0,
              post_unpack_func=lambda weights: received.extend(weights),
          )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:233: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f7850037430>
  group = <MagicMock id='140154723440480'>, src = 0
  post_unpack_func = <function test_packed_broadcast_consumer_single_tensor.<locals>.<lambda> at 0x7f78506d5620>
  buffer_size_bytes = 1073741824, num_buffers = 2
  
      def packed_broadcast_consumer(
          iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
          group: Any,
          src: int,
          post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
          num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
      ) -> None:
          """Consume packed tensors and unpack them into a list of tensors.
      
          Args:
              iterator: Iterator of parameter metadata. Returns (name, (shape, dtype))
              group: Process group (PyHcclCommunicator)
              src: Source rank (0 in current implementation)
              post_unpack_func: Function to apply to each list of (name, tensor) after
                               unpacking
              buffer_size_bytes: Size in bytes for each packed tensor buffer.
                                Both producer and consumer must use the same value.
              num_buffers: Number of buffers for double/triple buffering.
                          Both producer and consumer must use the same value.
          """
      
          def unpack_tensor(
              packed_tensor: torch.Tensor,
              names: list[str],
              shapes: list[list[int]],
              dtypes: list[torch.dtype],
              tensor_sizes: list[int],
          ) -> list[tuple[str, torch.Tensor]]:
              """Unpack a packed uint8 tensor into a list of typed tensors."""
              unpacked_tensors = packed_tensor.split(tensor_sizes)
              unpacked_list = [
                  (name, tensor.contiguous().view(dtype).view(*shape))
                  for name, shape, dtype, tensor in zip(names, shapes, dtypes, unpacked_tensors)
              ]
              return unpacked_list
      
          target_packed_tensor_size = buffer_size_bytes
      
          streams = [torch.npu.Stream() for _ in range(num_buffers)]
          default_stream = torch.npu.current_stream()
          buffer_idx = 0
      
          packing_tensor_meta_data: list[list[tuple[str, list[int], torch.dtype, int]]] = [[] for _ in range(num_buffers)]
          packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
  >       packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:137: ModuleNotFoundError
  __________ test_packed_broadcast_consumer_multiple_tensors_one_buffer __________
  
      def test_packed_broadcast_consumer_multiple_tensors_one_buffer():
          """Consumer unpacks multiple tensors from one packed buffer."""
          a = torch.full((4,), 1.0, dtype=torch.float32)
          b = torch.full((8,), 2.0, dtype=torch.float32)
          packed = torch.cat([a.view(torch.uint8).view(-1), b.view(torch.uint8).view(-1)])
          group = _make_group_mock()
          group.broadcast = MagicMock(side_effect=lambda t, **kw: t.copy_(packed))
      
          received: list[tuple[str, torch.Tensor]] = []
  >       packed_broadcast_consumer(
              iterator=iter([("a", ([4], torch.float32)), ("b", ([8], torch.float32))]),
              group=group,
              src=0,
              post_unpack_func=lambda weights: received.extend(weights),
          )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:257: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f78507d4370>
  group = <MagicMock id='140154715202784'>, src = 0
  post_unpack_func = <function test_packed_broadcast_consumer_multiple_tensors_one_buffer.<locals>.<lambda> at 0x7f78506d71a0>
  buffer_size_bytes = 1073741824, num_buffers = 2
  
      def packed_broadcast_consumer(
          iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
          group: Any,
          src: int,
          post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
          num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
      ) -> None:
          """Consume packed tensors and unpack them into a list of tensors.
      
          Args:
              iterator: Iterator of parameter metadata. Returns (name, (shape, dtype))
              group: Process group (PyHcclCommunicator)
              src: Source rank (0 in current implementation)
              post_unpack_func: Function to apply to each list of (name, tensor) after
                               unpacking
              buffer_size_bytes: Size in bytes for each packed tensor buffer.
                                Both producer and consumer must use the same value.
              num_buffers: Number of buffers for double/triple buffering.
                          Both producer and consumer must use the same value.
          """
      
          def unpack_tensor(
              packed_tensor: torch.Tensor,
              names: list[str],
              shapes: list[list[int]],
              dtypes: list[torch.dtype],
              tensor_sizes: list[int],
          ) -> list[tuple[str, torch.Tensor]]:
              """Unpack a packed uint8 tensor into a list of typed tensors."""
              unpacked_tensors = packed_tensor.split(tensor_sizes)
              unpacked_list = [
                  (name, tensor.contiguous().view(dtype).view(*shape))
                  for name, shape, dtype, tensor in zip(names, shapes, dtypes, unpacked_tensors)
              ]
              return unpacked_list
      
          target_packed_tensor_size = buffer_size_bytes
      
          streams = [torch.npu.Stream() for _ in range(num_buffers)]
          default_stream = torch.npu.current_stream()
          buffer_idx = 0
      
          packing_tensor_meta_data: list[list[tuple[str, list[int], torch.dtype, int]]] = [[] for _ in range(num_buffers)]
          packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
  >       packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:137: ModuleNotFoundError
  ________________ test_packed_broadcast_consumer_empty_iterator _________________
  
      def test_packed_broadcast_consumer_empty_iterator():
          """Consumer with an empty iterator triggers no broadcasts."""
          group = _make_group_mock()
          received: list = []
  >       packed_broadcast_consumer(
              iterator=iter([]),
              group=group,
              src=0,
              post_unpack_func=lambda weights: received.extend(weights),
          )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:275: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f7850035690>
  group = <MagicMock id='140154715205136'>, src = 0
  post_unpack_func = <function test_packed_broadcast_consumer_empty_iterator.<locals>.<lambda> at 0x7f785071ce00>
  buffer_size_bytes = 1073741824, num_buffers = 2
  
      def packed_broadcast_consumer(
          iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
          group: Any,
          src: int,
          post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
          num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
      ) -> None:
          """Consume packed tensors and unpack them into a list of tensors.
      
          Args:
              iterator: Iterator of parameter metadata. Returns (name, (shape, dtype))
              group: Process group (PyHcclCommunicator)
              src: Source rank (0 in current implementation)
              post_unpack_func: Function to apply to each list of (name, tensor) after
                               unpacking
              buffer_size_bytes: Size in bytes for each packed tensor buffer.
                                Both producer and consumer must use the same value.
              num_buffers: Number of buffers for double/triple buffering.
                          Both producer and consumer must use the same value.
          """
      
          def unpack_tensor(
              packed_tensor: torch.Tensor,
              names: list[str],
              shapes: list[list[int]],
              dtypes: list[torch.dtype],
              tensor_sizes: list[int],
          ) -> list[tuple[str, torch.Tensor]]:
              """Unpack a packed uint8 tensor into a list of typed tensors."""
              unpacked_tensors = packed_tensor.split(tensor_sizes)
              unpacked_list = [
                  (name, tensor.contiguous().view(dtype).view(*shape))
                  for name, shape, dtype, tensor in zip(names, shapes, dtypes, unpacked_tensors)
              ]
              return unpacked_list
      
          target_packed_tensor_size = buffer_size_bytes
      
          streams = [torch.npu.Stream() for _ in range(num_buffers)]
          default_stream = torch.npu.current_stream()
          buffer_idx = 0
      
          packing_tensor_meta_data: list[list[tuple[str, list[int], torch.dtype, int]]] = [[] for _ in range(num_buffers)]
          packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
  >       packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:137: ModuleNotFoundError
  ________________ test_packed_broadcast_consumer_passes_src_rank ________________
  
      def test_packed_broadcast_consumer_passes_src_rank():
          """``src`` is forwarded to ``group.broadcast``."""
          packed = torch.zeros(4, dtype=torch.uint8)
          group = _make_group_mock()
          group.broadcast = MagicMock(side_effect=lambda t, **kw: t.copy_(packed))
      
  >       packed_broadcast_consumer(
              iterator=iter([("w", ([1], torch.float32))]),
              group=group,
              src=2,
              post_unpack_func=lambda _: None,
          )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:291: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f785002b730>
  group = <MagicMock id='140154723443312'>, src = 2
  post_unpack_func = <function test_packed_broadcast_consumer_passes_src_rank.<locals>.<lambda> at 0x7f78506d5c60>
  buffer_size_bytes = 1073741824, num_buffers = 2
  
      def packed_broadcast_consumer(
          iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
          group: Any,
          src: int,
          post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
          num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
      ) -> None:
          """Consume packed tensors and unpack them into a list of tensors.
      
          Args:
              iterator: Iterator of parameter metadata. Returns (name, (shape, dtype))
              group: Process group (PyHcclCommunicator)
              src: Source rank (0 in current implementation)
              post_unpack_func: Function to apply to each list of (name, tensor) after
                               unpacking
              buffer_size_bytes: Size in bytes for each packed tensor buffer.
                                Both producer and consumer must use the same value.
              num_buffers: Number of buffers for double/triple buffering.
                          Both producer and consumer must use the same value.
          """
      
          def unpack_tensor(
              packed_tensor: torch.Tensor,
              names: list[str],
              shapes: list[list[int]],
              dtypes: list[torch.dtype],
              tensor_sizes: list[int],
          ) -> list[tuple[str, torch.Tensor]]:
              """Unpack a packed uint8 tensor into a list of typed tensors."""
              unpacked_tensors = packed_tensor.split(tensor_sizes)
              unpacked_list = [
                  (name, tensor.contiguous().view(dtype).view(*shape))
                  for name, shape, dtype, tensor in zip(names, shapes, dtypes, unpacked_tensors)
              ]
              return unpacked_list
      
          target_packed_tensor_size = buffer_size_bytes
      
          streams = [torch.npu.Stream() for _ in range(num_buffers)]
          default_stream = torch.npu.current_stream()
          buffer_idx = 0
      
          packing_tensor_meta_data: list[list[tuple[str, list[int], torch.dtype, int]]] = [[] for _ in range(num_buffers)]
          packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
  >       packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:137: ModuleNotFoundError
  ____________ test_packed_broadcast_consumer_handles_multiple_dtypes ____________
  
      def test_packed_broadcast_consumer_handles_multiple_dtypes():
          """Consumer correctly restores tensors of different dtypes from one buffer."""
          f16 = torch.tensor([1.5, 2.5], dtype=torch.float16)
          f32 = torch.tensor([3.0, 4.0], dtype=torch.float32)
          packed = torch.cat([f16.view(torch.uint8).view(-1), f32.view(torch.uint8).view(-1)])
          group = _make_group_mock()
          group.broadcast = MagicMock(side_effect=lambda t, **kw: t.copy_(packed))
      
          received: list[tuple[str, torch.Tensor]] = []
  >       packed_broadcast_consumer(
              iterator=iter([("f16", ([2], torch.float16)), ("f32", ([2], torch.float32))]),
              group=group,
              src=0,
              post_unpack_func=lambda weights: received.extend(weights),
          )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:309: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f7850032ef0>
  group = <MagicMock id='140154723440576'>, src = 0
  post_unpack_func = <function test_packed_broadcast_consumer_handles_multiple_dtypes.<locals>.<lambda> at 0x7f78506d6ac0>
  buffer_size_bytes = 1073741824, num_buffers = 2
  
      def packed_broadcast_consumer(
          iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
          group: Any,
          src: int,
          post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
          num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
      ) -> None:
          """Consume packed tensors and unpack them into a list of tensors.
      
          Args:
              iterator: Iterator of parameter metadata. Returns (name, (shape, dtype))
              group: Process group (PyHcclCommunicator)
              src: Source rank (0 in current implementation)
              post_unpack_func: Function to apply to each list of (name, tensor) after
                               unpacking
              buffer_size_bytes: Size in bytes for each packed tensor buffer.
                                Both producer and consumer must use the same value.
              num_buffers: Number of buffers for double/triple buffering.
                          Both producer and consumer must use the same value.
          """
      
          def unpack_tensor(
              packed_tensor: torch.Tensor,
              names: list[str],
              shapes: list[list[int]],
              dtypes: list[torch.dtype],
              tensor_sizes: list[int],
          ) -> list[tuple[str, torch.Tensor]]:
              """Unpack a packed uint8 tensor into a list of typed tensors."""
              unpacked_tensors = packed_tensor.split(tensor_sizes)
              unpacked_list = [
                  (name, tensor.contiguous().view(dtype).view(*shape))
                  for name, shape, dtype, tensor in zip(names, shapes, dtypes, unpacked_tensors)
              ]
              return unpacked_list
      
          target_packed_tensor_size = buffer_size_bytes
      
          streams = [torch.npu.Stream() for _ in range(num_buffers)]
          default_stream = torch.npu.current_stream()
          buffer_idx = 0
      
          packing_tensor_meta_data: list[list[tuple[str, list[int], torch.dtype, int]]] = [[] for _ in range(num_buffers)]
          packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
  >       packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:137: ModuleNotFoundError
  __________________ test_packed_npu_ipc_producer_single_chunk ___________________
  
      def test_packed_npu_ipc_producer_single_chunk():
          """A small tensor fits in one chunk; one dict is yielded."""
          patcher, args_sentinel = _install_fake_reduce_tensor()
          with patcher:
              tensor = torch.full((4,), 1.5, dtype=torch.float32)
  >           chunks = list(
                  packed_npu_ipc_producer(
                      iterator=iter([("w", tensor)]),
                      npu_uuid="node-0",
                      post_iter_func=lambda item: item[1],
                      buffer_size_bytes=DEFAULT_PACKED_BUFFER_SIZE_BYTES,
                  )
              )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:343: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f78508019c0>, npu_uuid = 'node-0'
  post_iter_func = <function test_packed_npu_ipc_producer_single_chunk.<locals>.<lambda> at 0x7f78506d5da0>
  buffer_size_bytes = 1073741824
  
      def packed_npu_ipc_producer(
          iterator: Iterator[tuple[str, torch.Tensor]],
          npu_uuid: str,
          post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
      ) -> Iterator[dict[str, Any]]:
          """Pack tensors into a reusable NPU IPC buffer and yield chunks.
      
          Allocates a single NPU buffer of ``buffer_size_bytes`` and registers
          it for IPC once via ``reduce_tensor``.  Each chunk's packed data is
          copied into this buffer before yielding, so only one IPC-shared
          allocation exists for the lifetime of the transfer.
      
          Args:
              iterator: Iterator of (name, tensor) pairs.
              npu_uuid: Physical NPU UUID string for this rank.
              post_iter_func: Applied to each (name, tensor) before packing.
              buffer_size_bytes: Exact capacity of the reusable IPC buffer.
          """
  >       ipc_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8, device="npu")
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:220: ModuleNotFoundError
  _________________ test_packed_npu_ipc_producer_empty_iterator __________________
  
      def test_packed_npu_ipc_producer_empty_iterator():
          """An empty iterator yields no chunks."""
          patcher, _ = _install_fake_reduce_tensor()
          with patcher:
  >           chunks = list(
                  packed_npu_ipc_producer(
                      iterator=iter([]),
                      npu_uuid="node-0",
                      post_iter_func=lambda item: item[1],
                      buffer_size_bytes=DEFAULT_PACKED_BUFFER_SIZE_BYTES,
                  )
              )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:365: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f78507f3400>, npu_uuid = 'node-0'
  post_iter_func = <function test_packed_npu_ipc_producer_empty_iterator.<locals>.<lambda> at 0x7f78506d7880>
  buffer_size_bytes = 1073741824
  
      def packed_npu_ipc_producer(
          iterator: Iterator[tuple[str, torch.Tensor]],
          npu_uuid: str,
          post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
      ) -> Iterator[dict[str, Any]]:
          """Pack tensors into a reusable NPU IPC buffer and yield chunks.
      
          Allocates a single NPU buffer of ``buffer_size_bytes`` and registers
          it for IPC once via ``reduce_tensor``.  Each chunk's packed data is
          copied into this buffer before yielding, so only one IPC-shared
          allocation exists for the lifetime of the transfer.
      
          Args:
              iterator: Iterator of (name, tensor) pairs.
              npu_uuid: Physical NPU UUID string for this rank.
              post_iter_func: Applied to each (name, tensor) before packing.
              buffer_size_bytes: Exact capacity of the reusable IPC buffer.
          """
  >       ipc_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8, device="npu")
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:220: ModuleNotFoundError
  __________ test_packed_npu_ipc_producer_splits_when_exceeding_buffer ___________
  
      def test_packed_npu_ipc_producer_splits_when_exceeding_buffer():
          """Tensors that don't fit together are split across multiple chunks."""
          patcher, args_sentinel = _install_fake_reduce_tensor()
          with patcher:
              tensors = [
                  ("a", torch.full((4,), 1.0, dtype=torch.float32)),  # 16 bytes
                  ("b", torch.full((4,), 2.0, dtype=torch.float32)),  # 16 bytes
              ]
  >           chunks = list(
                  packed_npu_ipc_producer(
                      iterator=iter(tensors),
                      npu_uuid="node-0",
                      post_iter_func=lambda item: item[1],
                      buffer_size_bytes=20,  # Only one 16-byte tensor fits per chunk
                  )
              )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:384: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f78507f1240>, npu_uuid = 'node-0'
  post_iter_func = <function test_packed_npu_ipc_producer_splits_when_exceeding_buffer.<locals>.<lambda> at 0x7f784ff64040>
  buffer_size_bytes = 20
  
      def packed_npu_ipc_producer(
          iterator: Iterator[tuple[str, torch.Tensor]],
          npu_uuid: str,
          post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
      ) -> Iterator[dict[str, Any]]:
          """Pack tensors into a reusable NPU IPC buffer and yield chunks.
      
          Allocates a single NPU buffer of ``buffer_size_bytes`` and registers
          it for IPC once via ``reduce_tensor``.  Each chunk's packed data is
          copied into this buffer before yielding, so only one IPC-shared
          allocation exists for the lifetime of the transfer.
      
          Args:
              iterator: Iterator of (name, tensor) pairs.
              npu_uuid: Physical NPU UUID string for this rank.
              post_iter_func: Applied to each (name, tensor) before packing.
              buffer_size_bytes: Exact capacity of the reusable IPC buffer.
          """
  >       ipc_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8, device="npu")
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:220: ModuleNotFoundError
  ____ test_packed_npu_ipc_producer_raises_when_single_tensor_exceeds_buffer _____
  
      def test_packed_npu_ipc_producer_raises_when_single_tensor_exceeds_buffer():
          """A single tensor larger than the buffer raises ``ValueError``."""
          patcher, _ = _install_fake_reduce_tensor()
          with patcher:
              big = torch.zeros(64, dtype=torch.float32)  # 256 bytes
              with pytest.raises(ValueError, match="exceeds buffer_size_bytes"):
  >               list(
                      packed_npu_ipc_producer(
                          iterator=iter([("big", big)]),
                          npu_uuid="node-0",
                          post_iter_func=lambda item: item[1],
                          buffer_size_bytes=128,
                      )
                  )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:410: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f7850800bb0>, npu_uuid = 'node-0'
  post_iter_func = <function test_packed_npu_ipc_producer_raises_when_single_tensor_exceeds_buffer.<locals>.<lambda> at 0x7f78506d7d80>
  buffer_size_bytes = 128
  
      def packed_npu_ipc_producer(
          iterator: Iterator[tuple[str, torch.Tensor]],
          npu_uuid: str,
          post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
      ) -> Iterator[dict[str, Any]]:
          """Pack tensors into a reusable NPU IPC buffer and yield chunks.
      
          Allocates a single NPU buffer of ``buffer_size_bytes`` and registers
          it for IPC once via ``reduce_tensor``.  Each chunk's packed data is
          copied into this buffer before yielding, so only one IPC-shared
          allocation exists for the lifetime of the transfer.
      
          Args:
              iterator: Iterator of (name, tensor) pairs.
              npu_uuid: Physical NPU UUID string for this rank.
              post_iter_func: Applied to each (name, tensor) before packing.
              buffer_size_bytes: Exact capacity of the reusable IPC buffer.
          """
  >       ipc_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8, device="npu")
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:220: ModuleNotFoundError
  ______________ test_packed_npu_ipc_producer_dtype_name_extraction ______________
  
      def test_packed_npu_ipc_producer_dtype_name_extraction():
          """``dtype_names`` strips the ``torch.`` prefix from ``str(dtype)``."""
          patcher, _ = _install_fake_reduce_tensor()
          with patcher:
              tensors = [
                  ("f16", torch.zeros(1, dtype=torch.float16)),
                  ("bf16", torch.zeros(1, dtype=torch.bfloat16)),
                  ("i64", torch.zeros(1, dtype=torch.int64)),
              ]
  >           chunks = list(
                  packed_npu_ipc_producer(
                      iterator=iter(tensors),
                      npu_uuid="node-0",
                      post_iter_func=lambda item: item[1],
                      buffer_size_bytes=DEFAULT_PACKED_BUFFER_SIZE_BYTES,
                  )
              )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:429: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f7851d22200>, npu_uuid = 'node-0'
  post_iter_func = <function test_packed_npu_ipc_producer_dtype_name_extraction.<locals>.<lambda> at 0x7f78506d6520>
  buffer_size_bytes = 1073741824
  
      def packed_npu_ipc_producer(
          iterator: Iterator[tuple[str, torch.Tensor]],
          npu_uuid: str,
          post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
      ) -> Iterator[dict[str, Any]]:
          """Pack tensors into a reusable NPU IPC buffer and yield chunks.
      
          Allocates a single NPU buffer of ``buffer_size_bytes`` and registers
          it for IPC once via ``reduce_tensor``.  Each chunk's packed data is
          copied into this buffer before yielding, so only one IPC-shared
          allocation exists for the lifetime of the transfer.
      
          Args:
              iterator: Iterator of (name, tensor) pairs.
              npu_uuid: Physical NPU UUID string for this rank.
              post_iter_func: Applied to each (name, tensor) before packing.
              buffer_size_bytes: Exact capacity of the reusable IPC buffer.
          """
  >       ipc_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8, device="npu")
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:220: ModuleNotFoundError
  _______ test_packed_broadcast_producer_post_iter_func_transforms_tensor ________
  
      def test_packed_broadcast_producer_post_iter_func_transforms_tensor():
          """``post_iter_func`` may transform the tensor before packing.
      
          The producer passes each ``(name, tensor)`` item through ``post_iter_func``
          and packs the returned tensor.  A non-identity transform (e.g. ``t * 2``)
          must be applied before the bytes hit the wire.
          """
          original = torch.full((4,), 1.0, dtype=torch.float32)
          group = _make_group_mock()
      
  >       packed_broadcast_producer(
              iterator=iter([("w", original)]),
              group=group,
              src=0,
              post_iter_func=lambda item: item[1] * 2,
          )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:585: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f78508121d0>
  group = <MagicMock id='140154723444944'>, src = 0
  post_iter_func = <function test_packed_broadcast_producer_post_iter_func_transforms_tensor.<locals>.<lambda> at 0x7f78506d6480>
  buffer_size_bytes = 1073741824, num_buffers = 2
  
      def packed_broadcast_producer(
          iterator: Iterator[tuple[str, torch.Tensor]],
          group: Any,
          src: int,
          post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
          num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
      ) -> None:
          """Broadcast tensors in a packed manner from trainer to workers.
      
          Args:
              iterator: Iterator of model parameters. Returns a tuple of (name, tensor)
              group: Process group (PyHcclCommunicator)
              src: Source rank (0 in current implementation)
              post_iter_func: Function to apply to each (name, tensor) pair before
                             packing, should return a tensor
              buffer_size_bytes: Size in bytes for each packed tensor buffer.
                                Both producer and consumer must use the same value.
              num_buffers: Number of buffers for double/triple buffering.
                          Both producer and consumer must use the same value.
          """
          target_packed_tensor_size = buffer_size_bytes
      
          streams = [torch.npu.Stream() for _ in range(num_buffers)]
          buffer_idx = 0
      
          packing_tensor_list: list[list[torch.Tensor]] = [[] for _ in range(num_buffers)]
          packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
  >       packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:46: ModuleNotFoundError
  _________ test_packed_broadcast_producer_handles_non_contiguous_tensor _________
  
      def test_packed_broadcast_producer_handles_non_contiguous_tensor():
          """The producer calls ``.contiguous()`` on each tensor before packing.
      
          A non-contiguous tensor (e.g. a transposed view) must be packed using its
          logical byte layout, not its strides.
          """
          base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
          non_contig = base.t()  # transpose → non-contiguous
          assert not non_contig.is_contiguous()
          group = _make_group_mock()
      
  >       packed_broadcast_producer(
              iterator=iter([("w", non_contig)]),
              group=group,
              src=0,
              post_iter_func=lambda item: item[1],
          )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:607: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f785002a1a0>
  group = <MagicMock id='140154723438416'>, src = 0
  post_iter_func = <function test_packed_broadcast_producer_handles_non_contiguous_tensor.<locals>.<lambda> at 0x7f784ff642c0>
  buffer_size_bytes = 1073741824, num_buffers = 2
  
      def packed_broadcast_producer(
          iterator: Iterator[tuple[str, torch.Tensor]],
          group: Any,
          src: int,
          post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
          num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
      ) -> None:
          """Broadcast tensors in a packed manner from trainer to workers.
      
          Args:
              iterator: Iterator of model parameters. Returns a tuple of (name, tensor)
              group: Process group (PyHcclCommunicator)
              src: Source rank (0 in current implementation)
              post_iter_func: Function to apply to each (name, tensor) pair before
                             packing, should return a tensor
              buffer_size_bytes: Size in bytes for each packed tensor buffer.
                                Both producer and consumer must use the same value.
              num_buffers: Number of buffers for double/triple buffering.
                          Both producer and consumer must use the same value.
          """
          target_packed_tensor_size = buffer_size_bytes
      
          streams = [torch.npu.Stream() for _ in range(num_buffers)]
          buffer_idx = 0
      
          packing_tensor_list: list[list[torch.Tensor]] = [[] for _ in range(num_buffers)]
          packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
  >       packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:46: ModuleNotFoundError
  __________ test_packed_broadcast_producer_tensor_exactly_fills_buffer __________
  
      def test_packed_broadcast_producer_tensor_exactly_fills_buffer():
          """A tensor whose byte size equals ``buffer_size_bytes`` is broadcast alone.
      
          The producer splits when ``packing_tensor_sizes > target_packed_tensor_size``
          (strict greater-than), so a tensor that exactly equals the threshold stays
          in the current buffer.
          """
          tensor = torch.full((4,), 3.0, dtype=torch.float32)  # 16 bytes
          group = _make_group_mock()
      
  >       packed_broadcast_producer(
              iterator=iter([("w", tensor)]),
              group=group,
              src=0,
              post_iter_func=lambda item: item[1],
              buffer_size_bytes=16,
          )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:628: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f78507d7a00>
  group = <MagicMock id='140154715147920'>, src = 0
  post_iter_func = <function test_packed_broadcast_producer_tensor_exactly_fills_buffer.<locals>.<lambda> at 0x7f78506d7600>
  buffer_size_bytes = 16, num_buffers = 2
  
      def packed_broadcast_producer(
          iterator: Iterator[tuple[str, torch.Tensor]],
          group: Any,
          src: int,
          post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
          num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
      ) -> None:
          """Broadcast tensors in a packed manner from trainer to workers.
      
          Args:
              iterator: Iterator of model parameters. Returns a tuple of (name, tensor)
              group: Process group (PyHcclCommunicator)
              src: Source rank (0 in current implementation)
              post_iter_func: Function to apply to each (name, tensor) pair before
                             packing, should return a tensor
              buffer_size_bytes: Size in bytes for each packed tensor buffer.
                                Both producer and consumer must use the same value.
              num_buffers: Number of buffers for double/triple buffering.
                          Both producer and consumer must use the same value.
          """
          target_packed_tensor_size = buffer_size_bytes
      
          streams = [torch.npu.Stream() for _ in range(num_buffers)]
          buffer_idx = 0
      
          packing_tensor_list: list[list[torch.Tensor]] = [[] for _ in range(num_buffers)]
          packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
  >       packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:46: ModuleNotFoundError
  _____ test_packed_broadcast_consumer_unpacks_multiple_dtypes_in_one_buffer _____
  
      def test_packed_broadcast_consumer_unpacks_multiple_dtypes_in_one_buffer():
          """Consumer restores fp16 and int32 tensors from the same packed buffer.
      
          Mixed dtypes in one buffer must each restore to the correct dtype and shape.
          """
          f16 = torch.tensor([1.5, 2.5], dtype=torch.float16)
          i32 = torch.tensor([10, 20, 30], dtype=torch.int32)
          packed = torch.cat([f16.view(torch.uint8).view(-1), i32.view(torch.uint8).view(-1)])
          group = _make_group_mock()
          group.broadcast = MagicMock(side_effect=lambda t, **kw: t.copy_(packed))
      
          received: list[tuple[str, torch.Tensor]] = []
  >       packed_broadcast_consumer(
              iterator=iter([("f16", ([2], torch.float16)), ("i32", ([3], torch.int32))]),
              group=group,
              src=0,
              post_unpack_func=lambda weights: received.extend(weights),
          )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:654: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f785002b970>
  group = <MagicMock id='140154723440528'>, src = 0
  post_unpack_func = <function test_packed_broadcast_consumer_unpacks_multiple_dtypes_in_one_buffer.<locals>.<lambda> at 0x7f78506d7920>
  buffer_size_bytes = 1073741824, num_buffers = 2
  
      def packed_broadcast_consumer(
          iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
          group: Any,
          src: int,
          post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
          num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
      ) -> None:
          """Consume packed tensors and unpack them into a list of tensors.
      
          Args:
              iterator: Iterator of parameter metadata. Returns (name, (shape, dtype))
              group: Process group (PyHcclCommunicator)
              src: Source rank (0 in current implementation)
              post_unpack_func: Function to apply to each list of (name, tensor) after
                               unpacking
              buffer_size_bytes: Size in bytes for each packed tensor buffer.
                                Both producer and consumer must use the same value.
              num_buffers: Number of buffers for double/triple buffering.
                          Both producer and consumer must use the same value.
          """
      
          def unpack_tensor(
              packed_tensor: torch.Tensor,
              names: list[str],
              shapes: list[list[int]],
              dtypes: list[torch.dtype],
              tensor_sizes: list[int],
          ) -> list[tuple[str, torch.Tensor]]:
              """Unpack a packed uint8 tensor into a list of typed tensors."""
              unpacked_tensors = packed_tensor.split(tensor_sizes)
              unpacked_list = [
                  (name, tensor.contiguous().view(dtype).view(*shape))
                  for name, shape, dtype, tensor in zip(names, shapes, dtypes, unpacked_tensors)
              ]
              return unpacked_list
      
          target_packed_tensor_size = buffer_size_bytes
      
          streams = [torch.npu.Stream() for _ in range(num_buffers)]
          default_stream = torch.npu.current_stream()
          buffer_idx = 0
      
          packing_tensor_meta_data: list[list[tuple[str, list[int], torch.dtype, int]]] = [[] for _ in range(num_buffers)]
          packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
  >       packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:137: ModuleNotFoundError
  ___________ test_packed_broadcast_consumer_restores_multi_dim_shape ____________
  
      def test_packed_broadcast_consumer_restores_multi_dim_shape():
          """Consumer restores a 2-D tensor from its packed 1-D representation.
      
          ``view(*shape)`` must reconstruct the original multi-dimensional layout.
          """
          original = torch.arange(6, dtype=torch.float32).reshape(2, 3)
          packed = original.view(torch.uint8).view(-1).clone()
          group = _make_group_mock()
          group.broadcast = MagicMock(side_effect=lambda t, **kw: t.copy_(packed))
      
          received: list[tuple[str, torch.Tensor]] = []
  >       packed_broadcast_consumer(
              iterator=iter([("w", ([2, 3], torch.float32))]),
              group=group,
              src=0,
              post_unpack_func=lambda weights: received.extend(weights),
          )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:678: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f7850803b80>
  group = <MagicMock id='140154715196160'>, src = 0
  post_unpack_func = <function test_packed_broadcast_consumer_restores_multi_dim_shape.<locals>.<lambda> at 0x7f784ff64360>
  buffer_size_bytes = 1073741824, num_buffers = 2
  
      def packed_broadcast_consumer(
          iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
          group: Any,
          src: int,
          post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
          num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
      ) -> None:
          """Consume packed tensors and unpack them into a list of tensors.
      
          Args:
              iterator: Iterator of parameter metadata. Returns (name, (shape, dtype))
              group: Process group (PyHcclCommunicator)
              src: Source rank (0 in current implementation)
              post_unpack_func: Function to apply to each list of (name, tensor) after
                               unpacking
              buffer_size_bytes: Size in bytes for each packed tensor buffer.
                                Both producer and consumer must use the same value.
              num_buffers: Number of buffers for double/triple buffering.
                          Both producer and consumer must use the same value.
          """
      
          def unpack_tensor(
              packed_tensor: torch.Tensor,
              names: list[str],
              shapes: list[list[int]],
              dtypes: list[torch.dtype],
              tensor_sizes: list[int],
          ) -> list[tuple[str, torch.Tensor]]:
              """Unpack a packed uint8 tensor into a list of typed tensors."""
              unpacked_tensors = packed_tensor.split(tensor_sizes)
              unpacked_list = [
                  (name, tensor.contiguous().view(dtype).view(*shape))
                  for name, shape, dtype, tensor in zip(names, shapes, dtypes, unpacked_tensors)
              ]
              return unpacked_list
      
          target_packed_tensor_size = buffer_size_bytes
      
          streams = [torch.npu.Stream() for _ in range(num_buffers)]
          default_stream = torch.npu.current_stream()
          buffer_idx = 0
      
          packing_tensor_meta_data: list[list[tuple[str, list[int], torch.dtype, int]]] = [[] for _ in range(num_buffers)]
          packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
  >       packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:137: ModuleNotFoundError
  _______________ test_packed_broadcast_consumer_handles_bfloat16 ________________
  
      def test_packed_broadcast_consumer_handles_bfloat16():
          """Consumer correctly restores bfloat16 tensors.
      
          bfloat16 has 2-byte elements; the unpacker must use the correct itemsize
          when slicing and the correct dtype when viewing.
          """
          original = torch.tensor([1.5, -0.5, 2.25], dtype=torch.bfloat16)
          packed = original.view(torch.uint8).view(-1).clone()
          group = _make_group_mock()
          group.broadcast = MagicMock(side_effect=lambda t, **kw: t.copy_(packed))
      
          received: list[tuple[str, torch.Tensor]] = []
  >       packed_broadcast_consumer(
              iterator=iter([("w", ([3], torch.bfloat16))]),
              group=group,
              src=0,
              post_unpack_func=lambda weights: received.extend(weights),
          )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:701: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f78507d4070>
  group = <MagicMock id='140154715156368'>, src = 0
  post_unpack_func = <function test_packed_broadcast_consumer_handles_bfloat16.<locals>.<lambda> at 0x7f784ff64b80>
  buffer_size_bytes = 1073741824, num_buffers = 2
  
      def packed_broadcast_consumer(
          iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
          group: Any,
          src: int,
          post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
          num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
      ) -> None:
          """Consume packed tensors and unpack them into a list of tensors.
      
          Args:
              iterator: Iterator of parameter metadata. Returns (name, (shape, dtype))
              group: Process group (PyHcclCommunicator)
              src: Source rank (0 in current implementation)
              post_unpack_func: Function to apply to each list of (name, tensor) after
                               unpacking
              buffer_size_bytes: Size in bytes for each packed tensor buffer.
                                Both producer and consumer must use the same value.
              num_buffers: Number of buffers for double/triple buffering.
                          Both producer and consumer must use the same value.
          """
      
          def unpack_tensor(
              packed_tensor: torch.Tensor,
              names: list[str],
              shapes: list[list[int]],
              dtypes: list[torch.dtype],
              tensor_sizes: list[int],
          ) -> list[tuple[str, torch.Tensor]]:
              """Unpack a packed uint8 tensor into a list of typed tensors."""
              unpacked_tensors = packed_tensor.split(tensor_sizes)
              unpacked_list = [
                  (name, tensor.contiguous().view(dtype).view(*shape))
                  for name, shape, dtype, tensor in zip(names, shapes, dtypes, unpacked_tensors)
              ]
              return unpacked_list
      
          target_packed_tensor_size = buffer_size_bytes
      
          streams = [torch.npu.Stream() for _ in range(num_buffers)]
          default_stream = torch.npu.current_stream()
          buffer_idx = 0
      
          packing_tensor_meta_data: list[list[tuple[str, list[int], torch.dtype, int]]] = [[] for _ in range(num_buffers)]
          packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
  >       packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:137: ModuleNotFoundError
  ________________ test_packed_broadcast_roundtrip_single_tensor _________________
  
      def test_packed_broadcast_roundtrip_single_tensor():
          """Producer → consumer roundtrip preserves tensor bytes and metadata."""
          original = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
          group = _make_group_mock()
      
          captured: list[torch.Tensor] = []
          group.broadcast = MagicMock(side_effect=lambda t, **kw: captured.append(t))
  >       packed_broadcast_producer(
              iterator=iter([("w", original)]),
              group=group,
              src=0,
              post_iter_func=lambda item: item[1],
          )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:724: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f785059c580>
  group = <MagicMock id='140154723198752'>, src = 0
  post_iter_func = <function test_packed_broadcast_roundtrip_single_tensor.<locals>.<lambda> at 0x7f784ff64720>
  buffer_size_bytes = 1073741824, num_buffers = 2
  
      def packed_broadcast_producer(
          iterator: Iterator[tuple[str, torch.Tensor]],
          group: Any,
          src: int,
          post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
          num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
      ) -> None:
          """Broadcast tensors in a packed manner from trainer to workers.
      
          Args:
              iterator: Iterator of model parameters. Returns a tuple of (name, tensor)
              group: Process group (PyHcclCommunicator)
              src: Source rank (0 in current implementation)
              post_iter_func: Function to apply to each (name, tensor) pair before
                             packing, should return a tensor
              buffer_size_bytes: Size in bytes for each packed tensor buffer.
                                Both producer and consumer must use the same value.
              num_buffers: Number of buffers for double/triple buffering.
                          Both producer and consumer must use the same value.
          """
          target_packed_tensor_size = buffer_size_bytes
      
          streams = [torch.npu.Stream() for _ in range(num_buffers)]
          buffer_idx = 0
      
          packing_tensor_list: list[list[torch.Tensor]] = [[] for _ in range(num_buffers)]
          packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
  >       packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:46: ModuleNotFoundError
  ________ test_packed_npu_ipc_producer_post_iter_func_transforms_tensor _________
  
      def test_packed_npu_ipc_producer_post_iter_func_transforms_tensor():
          """``post_iter_func`` may transform the tensor before packing in IPC mode."""
          patcher, _ = _install_fake_reduce_tensor()
          with patcher:
              original = torch.full((4,), 1.0, dtype=torch.float32)
  >           chunks = list(
                  packed_npu_ipc_producer(
                      iterator=iter([("w", original)]),
                      npu_uuid="node-0",
                      post_iter_func=lambda item: item[1] * 3,
                      buffer_size_bytes=DEFAULT_PACKED_BUFFER_SIZE_BYTES,
                  )
              )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:754: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f785059d5d0>, npu_uuid = 'node-0'
  post_iter_func = <function test_packed_npu_ipc_producer_post_iter_func_transforms_tensor.<locals>.<lambda> at 0x7f784ff64fe0>
  buffer_size_bytes = 1073741824
  
      def packed_npu_ipc_producer(
          iterator: Iterator[tuple[str, torch.Tensor]],
          npu_uuid: str,
          post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
      ) -> Iterator[dict[str, Any]]:
          """Pack tensors into a reusable NPU IPC buffer and yield chunks.
      
          Allocates a single NPU buffer of ``buffer_size_bytes`` and registers
          it for IPC once via ``reduce_tensor``.  Each chunk's packed data is
          copied into this buffer before yielding, so only one IPC-shared
          allocation exists for the lifetime of the transfer.
      
          Args:
              iterator: Iterator of (name, tensor) pairs.
              npu_uuid: Physical NPU UUID string for this rank.
              post_iter_func: Applied to each (name, tensor) before packing.
              buffer_size_bytes: Exact capacity of the reusable IPC buffer.
          """
  >       ipc_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8, device="npu")
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:220: ModuleNotFoundError
  _______ test_packed_npu_ipc_producer_handles_multiple_tensors_one_chunk ________
  
      def test_packed_npu_ipc_producer_handles_multiple_tensors_one_chunk():
          """Multiple small tensors fit in one IPC chunk and are yielded together."""
          patcher, _ = _install_fake_reduce_tensor()
          with patcher:
              a = torch.full((2,), 1.0, dtype=torch.float32)
              b = torch.full((3,), 2.0, dtype=torch.float32)
  >           chunks = list(
                  packed_npu_ipc_producer(
                      iterator=iter([("a", a), ("b", b)]),
                      npu_uuid="node-0",
                      post_iter_func=lambda item: item[1],
                      buffer_size_bytes=DEFAULT_PACKED_BUFFER_SIZE_BYTES,
                  )
              )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:775: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f7850789510>, npu_uuid = 'node-0'
  post_iter_func = <function test_packed_npu_ipc_producer_handles_multiple_tensors_one_chunk.<locals>.<lambda> at 0x7f784ff65620>
  buffer_size_bytes = 1073741824
  
      def packed_npu_ipc_producer(
          iterator: Iterator[tuple[str, torch.Tensor]],
          npu_uuid: str,
          post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
      ) -> Iterator[dict[str, Any]]:
          """Pack tensors into a reusable NPU IPC buffer and yield chunks.
      
          Allocates a single NPU buffer of ``buffer_size_bytes`` and registers
          it for IPC once via ``reduce_tensor``.  Each chunk's packed data is
          copied into this buffer before yielding, so only one IPC-shared
          allocation exists for the lifetime of the transfer.
      
          Args:
              iterator: Iterator of (name, tensor) pairs.
              npu_uuid: Physical NPU UUID string for this rank.
              post_iter_func: Applied to each (name, tensor) before packing.
              buffer_size_bytes: Exact capacity of the reusable IPC buffer.
          """
  >       ipc_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8, device="npu")
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:220: ModuleNotFoundError
  _________ test_packed_npu_ipc_producer_ipc_handle_key_matches_npu_uuid _________
  
      def test_packed_npu_ipc_producer_ipc_handle_key_matches_npu_uuid():
          """The ``ipc_handle`` dict key equals the ``npu_uuid`` argument.
      
          The consumer looks up the handle by its own UUID, so the producer must
          store it under the UUID it was given.
          """
          patcher, args_sentinel = _install_fake_reduce_tensor()
          with patcher:
  >           chunks = list(
                  packed_npu_ipc_producer(
                      iterator=iter([("w", torch.zeros(4, dtype=torch.float32))]),
                      npu_uuid="physical-npu-7",
                      post_iter_func=lambda item: item[1],
                      buffer_size_bytes=DEFAULT_PACKED_BUFFER_SIZE_BYTES,
                  )
              )
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py:798: 
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  
  iterator = <list_iterator object at 0x7f785059f340>, npu_uuid = 'physical-npu-7'
  post_iter_func = <function test_packed_npu_ipc_producer_ipc_handle_key_matches_npu_uuid.<locals>.<lambda> at 0x7f78506d5c60>
  buffer_size_bytes = 1073741824
  
      def packed_npu_ipc_producer(
          iterator: Iterator[tuple[str, torch.Tensor]],
          npu_uuid: str,
          post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
          buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
      ) -> Iterator[dict[str, Any]]:
          """Pack tensors into a reusable NPU IPC buffer and yield chunks.
      
          Allocates a single NPU buffer of ``buffer_size_bytes`` and registers
          it for IPC once via ``reduce_tensor``.  Each chunk's packed data is
          copied into this buffer before yielding, so only one IPC-shared
          allocation exists for the lifetime of the transfer.
      
          Args:
              iterator: Iterator of (name, tensor) pairs.
              npu_uuid: Physical NPU UUID string for this rank.
              post_iter_func: Applied to each (name, tensor) before packing.
              buffer_size_bytes: Exact capacity of the reusable IPC buffer.
          """
  >       ipc_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8, device="npu")
  E       ModuleNotFoundError: No module named 'torch.npu'
  
  vllm_ascend/distributed/weight_transfer/packed_tensor.py:220: ModuleNotFoundError
  =============================== warnings summary ===============================
  ../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch/jit/_script.py:362: 14 warnings
    /usr/local/python3.12.13/lib/python3.12/site-packages/torch/jit/_script.py:362: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
      warnings.warn(
  
  tests/ut/test_compressed_prefix_cache.py:26
    /__w/vllm-ascend/vllm-ascend/tests/ut/test_compressed_prefix_cache.py:26: PytestUnknownMarkWarning: Unknown pytest.mark.cpu_test - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
      pytestmark = pytest.mark.cpu_test
  
  tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_npu_ipc_consumer_clones_slices
    /__w/vllm-ascend/vllm-ascend/tests/ut/distributed/weight_transfer/test_packed_tensor.py:546: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      assert weights[0][1].storage().data_ptr() != weights[1][1].storage().data_ptr()
  
  -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
  =========================== short test summary info ============================
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_broadcast_producer_single_tensor - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_broadcast_producer_multiple_tensors_one_buffer - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_broadcast_producer_splits_when_exceeding_buffer - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_broadcast_producer_empty_iterator - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_broadcast_producer_uses_num_buffers_to_rotate - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_broadcast_producer_passes_src_rank - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_broadcast_consumer_single_tensor - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_broadcast_consumer_multiple_tensors_one_buffer - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_broadcast_consumer_empty_iterator - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_broadcast_consumer_passes_src_rank - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_broadcast_consumer_handles_multiple_dtypes - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_npu_ipc_producer_single_chunk - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_npu_ipc_producer_empty_iterator - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_npu_ipc_producer_splits_when_exceeding_buffer - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_npu_ipc_producer_raises_when_single_tensor_exceeds_buffer - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_npu_ipc_producer_dtype_name_extraction - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_broadcast_producer_post_iter_func_transforms_tensor - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_broadcast_producer_handles_non_contiguous_tensor - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_broadcast_producer_tensor_exactly_fills_buffer - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_broadcast_consumer_unpacks_multiple_dtypes_in_one_buffer - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_broadcast_consumer_restores_multi_dim_shape - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_broadcast_consumer_handles_bfloat16 - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_broadcast_roundtrip_single_tensor - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_npu_ipc_producer_post_iter_func_transforms_tensor - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_npu_ipc_producer_handles_multiple_tensors_one_chunk - ModuleNotFoundError: No module named 'torch.npu'
  FAILED tests/ut/distributed/weight_transfer/test_packed_tensor.py::test_packed_npu_ipc_producer_ipc_handle_key_matches_npu_uuid - ModuleNotFoundError: No module named 'torch.npu'
  ========== 26 failed, 2233 passed, 22 skipped, 16 warnings in 52.04s ===========
=== TEST SUMMARY ===
  FAILED: cpu-ut (185 targets)
    log: /__w/_temp/selected-tests-cpu-0card/1-cpu-ut.log
=== FAILED TEST LOGS ===
cpu-ut (185 targets) failure log
Error: Error: failed to run script step: Error: command terminated with non-zero exit code: command terminated with exit code 2
Error: Process completed with exit code 1.
Error: Executing the custom container implementation failed. Please contact your self hosted runner administrator.