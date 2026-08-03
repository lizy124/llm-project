# ASCEND_RT_VISIBLE_DEVICES 变化原因梳理

是的，按你给的启动脚本和这两个仓库代码，结论可以先说清楚：

`ASCEND_RT_VISIBLE_DEVICES` 在 `vllm-ascend` 的 `NPUWorker.init_device()` 里打印出来发生变化，代码层面能证明的来源主要不是 `worker.py` 自己改的，也不是 `torch.npu.set_device()` 改的。因为你的日志打印发生在 `torch.npu.set_device()` 之前。

关键代码在：

`E:/lizy/code/vllm-project/vllm-ascend/vllm_ascend/worker/worker.py:454`

```python
def init_device(self):
    logger.info(f"***********************ASCEND_RT_VISIBLE_DEVICES***********************{os.getenv('ASCEND_RT_VISIBLE_DEVICES')}")
    self.device = self._init_device()
```

而真正 `torch.npu.set_device()` 在后面的 `_init_device()` 里：

`E:/lizy/code/vllm-project/vllm-ascend/vllm_ascend/worker/worker.py:397`

```python
def _init_device(self):
    device = torch.device(f"npu:{self.local_rank}")
    torch.npu.set_device(device)
```

所以：如果你在 `init_device()` 第一行日志里已经看到 `ASCEND_RT_VISIBLE_DEVICES` 有值或变化，那么这个值一定是在进入 `NPUWorker.init_device()` 之前就已经存在于当前 Python worker 进程的 `os.environ` 里了。

代码链路如下。

## 1. `vllm-ascend` 本身有没有在 `init_device()` 前改它？

我查到的生产代码里，`vllm-ascend` 主要是声明和读取它，没有找到在 `NPUWorker.init_device()` 前直接执行：

```python
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ...
```

的生产路径。

`vllm-ascend` 把它注册成 NPU 平台的 device-control env：

`E:/lizy/code/vllm-project/vllm-ascend/vllm_ascend/platform.py:126`

```python
class NPUPlatform(Platform):
    ray_device_key: str = "NPU"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"
    ray_noset_device_env_vars: list[str] = [
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    ]
```

这说明上游 vLLM 的通用平台逻辑会把 `ASCEND_RT_VISIBLE_DEVICES` 当作 NPU 的“可见设备控制变量”来读。

但是这个声明本身不设置它。

另外，`vllm-ascend` 有读取它的地方，例如 A5 本地通信资源：

`E:/lizy/code/vllm-project/vllm-ascend/vllm_ascend/utils.py:518`

```python
visible_devices = os.getenv("ASCEND_RT_VISIBLE_DEVICES")
if visible_devices is None:
    from vllm_ascend.cpu_binding import DeviceInfo

    devices = sorted([int(x) for x in DeviceInfo.get_npu_map_info()])
else:
    devices = [int(x) for x in visible_devices.split(",") if x.strip()]
```

还有模块 import 时读取一次：

`E:/lizy/code/vllm-project/vllm-ascend/vllm_ascend/cpu_binding.py:19`

```python
ASCEND_RT_VISIBLE_DEVICES = os.getenv("ASCEND_RT_VISIBLE_DEVICES")
```

这些都是读，不是写。

所以仅从 `vllm-ascend` 生产代码看，不能证明它在 `worker.py:init_device()` 前主动改了 `ASCEND_RT_VISIBLE_DEVICES`。

## 2. 普通 multiproc / uniproc 路径下，谁会在 `init_device()` 前设置它？

如果不是 Ray backend，核心调用链是这样的。

### uniproc 路径

`E:/lizy/code/vllm-project/vllm/vllm/v1/executor/uniproc_executor.py:46`

```python
self.driver_worker.init_worker(all_kwargs=[kwargs])
self.driver_worker.init_device()
```

然后：

`E:/lizy/code/vllm-project/vllm/vllm/v1/worker/worker_base.py:327`

```python
def init_device(self):
    assert self.vllm_config is not None
    with set_current_vllm_config(self.vllm_config):
        self.worker.init_device()
```

最后进入你的：

`E:/lizy/code/vllm-project/vllm-ascend/vllm_ascend/worker/worker.py:454`

```python
logger.info(...)
self.device = self._init_device()
```

这条路径里，没有看到 vLLM 在 `init_device()` 前写 `ASCEND_RT_VISIBLE_DEVICES`。

### multiproc 路径

多进程 worker 创建后：

`E:/lizy/code/vllm-project/vllm/vllm/v1/executor/multiproc_executor.py:593`

```python
wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)
...
wrapper.init_worker(all_kwargs)
self.worker = wrapper
...
self.worker.init_device()
```

`init_worker()` 里会创建真实 worker：

`E:/lizy/code/vllm-project/vllm/vllm/v1/worker/worker_base.py:317`

```python
with set_current_vllm_config(self.vllm_config):
    self.worker = worker_class(**kwargs)
```

然后才调用：

`E:/lizy/code/vllm-project/vllm/vllm/v1/executor/multiproc_executor.py:626`

```python
self.worker.init_device()
```

这条路径也没有看到 vLLM 在 `init_device()` 前写 `ASCEND_RT_VISIBLE_DEVICES`。

所以如果你当前是普通 multiproc，本地 16 卡这类启动方式，那么 `worker.py:init_device()` 里看到的值，大概率来自 worker 进程继承的环境，而不是 vLLM/vllm-ascend 在这段链路里写的。

这里不是猜测，是代码链路上没有写入点。

## 3. Ray 路径下，确实存在“进入 `init_device()` 前传递/设置 env”的代码

Ray 是最需要重点看的地方，因为这里有明确的 worker env 传播逻辑。

### 3.1 RayExecutor 旧路径

vLLM 会先设置 Ray 的 no-set 变量，目的是让 Ray 不自动设置可见设备变量。

`E:/lizy/code/vllm-project/vllm/vllm/v1/executor/ray_executor.py:131`

```python
def _update_noset_device_env_vars(self, ray_remote_kwargs):
    runtime_env = ray_remote_kwargs.setdefault("runtime_env", {})
    env_vars = runtime_env.setdefault("env_vars", {})
    env_vars.update(
        {env_var: "1" for env_var in current_platform.ray_noset_device_env_vars}
    )
```

因为 `vllm-ascend` 的 `NPUPlatform.ray_noset_device_env_vars` 是：

```python
["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"]
```

所以 Ray 路径会设置：

```bash
RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
```

注意：这不是设置 `ASCEND_RT_VISIBLE_DEVICES`，而是告诉 Ray 不要自动设置它。

然后旧 Ray executor 会把 driver 侧某些环境变量复制到 worker：

`E:/lizy/code/vllm-project/vllm/vllm/v1/executor/ray_executor.py:309`

```python
env_vars_to_copy = get_env_vars_to_copy(
    exclude_vars=WORKER_SPECIFIC_ENV_VARS,
    additional_vars=set(current_platform.additional_env_vars),
    destination="workers",
)
...
for name in env_vars_to_copy:
    if name in os.environ:
        args[name] = os.environ[name]
```

之后在 worker 里执行：

`E:/lizy/code/vllm-project/vllm/vllm/v1/worker/worker_base.py:222`

```python
def update_environment_variables(self, envs_list: list[dict[str, str]]) -> None:
    envs = envs_list[self.rpc_rank]
    update_environment_variables(envs)
```

实际覆盖在：

`E:/lizy/code/vllm-project/vllm/vllm/utils/system_utils.py:34`

```python
def update_environment_variables(envs_dict: dict[str, str]):
    for k, v in envs_dict.items():
        if k in os.environ and os.environ[k] != v:
            logger.warning(
                "Overwriting environment variable %s from '%s' to '%s'",
                k,
                os.environ[k],
                v,
            )
        os.environ[k] = v
```

默认 `get_env_vars_to_copy()` 的前缀是：

`E:/lizy/code/vllm-project/vllm/vllm/ray/ray_env.py:36`

```python
DEFAULT_ENV_VAR_PREFIXES: set[str] = {
    "VLLM_",
    "LMCACHE_",
    "NCCL_",
    "UCX_",
    "HF_",
    "HUGGING_FACE_",
}
```

默认不包含 `ASCEND_`。

所以旧 RayExecutor 下：

- 默认不会复制 `ASCEND_RT_VISIBLE_DEVICES`
- 但如果你通过 `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=ASCEND_RT_VISIBLE_DEVICES` 或类似配置把它加进去，那么它会在 `init_device()` 前被 `update_environment_variables()` 写入 worker 进程

调用顺序是：

```text
RayExecutor._init_workers_ray
  -> _update_noset_device_env_vars()
  -> get_env_vars_to_copy()
  -> collective_rpc("update_environment_variables")
      -> WorkerWrapperBase.update_environment_variables()
          -> update_environment_variables()
  -> collective_rpc("init_worker")
  -> collective_rpc("init_device")
      -> NPUWorker.init_device()
```

所以旧 Ray 路径下，能证明的设置点是：

`vllm/utils/system_utils.py:update_environment_variables()`。

### 3.2 RayExecutorV2 路径更关键：它会把 driver env 传给 worker

RayExecutorV2 有更直接的代码。

它先构造 Ray runtime env：

`E:/lizy/code/vllm-project/vllm/vllm/v1/executor/ray_executor_v2.py:234`

```python
env_vars = runtime_env.setdefault("env_vars", {})
env_vars.update({v: "1" for v in current_platform.ray_noset_device_env_vars})
```

NPU 下就是：

```bash
RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
```

然后它收集 driver 侧环境变量：

`E:/lizy/code/vllm-project/vllm/vllm/v1/executor/ray_executor_v2.py:331`

```python
self.driver_env_vars = get_driver_env_vars(
    worker_specific_vars=WORKER_SPECIFIC_ENV_VARS,
)
```

`get_driver_env_vars()` 是：

`E:/lizy/code/vllm-project/vllm/vllm/v1/executor/ray_env_utils.py:8`

```python
def get_driver_env_vars(
    worker_specific_vars: set[str],
) -> dict[str, str]:
    exclude_vars = worker_specific_vars | RAY_NON_CARRY_OVER_ENV_VARS

    return {key: value for key, value in os.environ.items() if key not in exclude_vars}
```

关键点在 `WORKER_SPECIFIC_ENV_VARS`：

`E:/lizy/code/vllm-project/vllm/vllm/v1/executor/ray_utils.py:31`

```python
WORKER_SPECIFIC_ENV_VARS: set[str] = {
    "VLLM_HOST_IP",
    "VLLM_HOST_PORT",
    "VLLM_NIXL_SIDE_CHANNEL_HOST",
    "LOCAL_RANK",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
}
```

这里排除了：

```text
CUDA_VISIBLE_DEVICES
HIP_VISIBLE_DEVICES
ROCR_VISIBLE_DEVICES
```

但是没有排除：

```text
ASCEND_RT_VISIBLE_DEVICES
```

所以只要 RayExecutorV2 的 driver 进程 `os.environ` 里有 `ASCEND_RT_VISIBLE_DEVICES`，它就会被放进 `driver_env_vars`。

然后 worker 初始化时：

`E:/lizy/code/vllm-project/vllm/vllm/v1/executor/ray_executor_v2.py:138`

```python
def initialize_worker(
    self,
    local_rank: int,
    env_vars: dict[str, str],
    driver_env_vars: dict[str, str] | None = None,
    assigned_physical_gpu_ids: list[int] | None = None,
) -> None:
    if driver_env_vars:
        for key, value in driver_env_vars.items():
            os.environ.setdefault(key, value)
    for key, value in env_vars.items():
        os.environ[key] = value
```

也就是说：

```python
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", driver_value)
```

会发生在真正 `WorkerProc.__init__()` 前：

`E:/lizy/code/vllm-project/vllm/vllm/v1/executor/ray_executor_v2.py:165`

```python
self.local_rank = local_rank
super().__init__(
    local_rank=local_rank,
    **self._init_kwargs,
)
```

而 `WorkerProc.__init__()` 里才会调用 `init_device()`：

`E:/lizy/code/vllm-project/vllm/vllm/v1/executor/multiproc_executor.py:625`

```python
self.worker.init_device()
```

RayExecutorV2 的完整链路是：

```text
RayExecutorV2._init_executor()
  -> self.driver_env_vars = get_driver_env_vars(...)
      -> 没有排除 ASCEND_RT_VISIBLE_DEVICES
  -> 创建 RayWorkerProc actor
  -> actor.get_node_and_physical_gpu_ids()
  -> actor.initialize_worker(
         local_rank,
         worker_env_vars,
         self.driver_env_vars,
         assigned_physical_gpu_ids=...
     )
      -> os.environ.setdefault(key, value)
         这里可能设置 ASCEND_RT_VISIBLE_DEVICES
      -> super().__init__(...)
          -> WorkerProc.__init__()
              -> wrapper.init_worker(...)
              -> self.worker.init_device()
                  -> NPUWorker.init_device()
```

所以如果你用了 RayExecutorV2，那么代码层面可以明确解释：

`ASCEND_RT_VISIBLE_DEVICES` 是从 driver 进程环境变量通过 `driver_env_vars` 传到 Ray worker actor 的。

注意它用的是 `setdefault()`：

```python
os.environ.setdefault(key, value)
```

这意味着：

- 如果 Ray worker 进程里本来没有 `ASCEND_RT_VISIBLE_DEVICES`，它会被 driver 的值填上
- 如果 Ray worker 进程里已经有这个变量，它不会覆盖

## 4. vLLM 会读 `ASCEND_RT_VISIBLE_DEVICES` 做设备映射，但不是设置它

由于 `vllm-ascend` 声明了：

```python
device_control_env_var = "ASCEND_RT_VISIBLE_DEVICES"
```

上游 vLLM 的通用 platform 逻辑会读它。

例如 `--device-ids` 解析：

`E:/lizy/code/vllm-project/vllm/vllm/engine/arg_utils.py:1735`

```python
cvd = getattr(
    envs,
    current_platform.device_control_env_var,
    os.environ.get(current_platform.device_control_env_var),
)
if cvd:
    cvd_ids = [
        current_platform.device_control_id_to_physical_device_id(x)
        for x in cvd.split(",")
    ]
```

NPU 平台下，这里的：

```python
current_platform.device_control_env_var
```

就是：

```text
ASCEND_RT_VISIBLE_DEVICES
```

所以这里会读取它。

还有平台通用逻辑：

`E:/lizy/code/vllm-project/vllm/vllm/platforms/interface.py:275`

```python
if (
    cls.device_control_env_var in os.environ
    and os.environ[cls.device_control_env_var] != ""
):
    device_ids = os.environ[cls.device_control_env_var].split(",")
    physical_device_id = device_ids[device_id]
    return cls.device_control_id_to_physical_device_id(physical_device_id)
else:
    return device_id
```

以及：

`E:/lizy/code/vllm-project/vllm/vllm/platforms/interface.py:305`

```python
physical_device_id = cls.device_id_to_physical_device_id(device_id)
device_control_env = os.environ.get(cls.device_control_env_var, "")
if not device_control_env:
    return physical_device_id
...
return visible_physical_device_ids.index(physical_device_id)
```

这些代码会根据 `ASCEND_RT_VISIBLE_DEVICES` 做 logical device 到 physical/visible device 的映射。

但是这些地方也是读，不是写。

## 5. 你的启动脚本里 `unset` + `readonly` 不等于“worker 进程不能设置它”

你的脚本里是：

```bash
unset ASCEND_RT_VISIBLE_DEVICES
readonly ASCEND_RT_VISIBLE_DEVICES
```

这有几个重要点：

1. `unset ASCEND_RT_VISIBLE_DEVICES` 会把当前 shell 里的变量去掉。
2. `readonly ASCEND_RT_VISIBLE_DEVICES` 是 shell 自己的 readonly 属性。
3. 这个 readonly 属性不是 OS 级别限制。
4. Python 子进程内部依然可以执行：

```python
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "..."
```

5. 多进程/Ray actor 进程里设置自己的 `os.environ`，不会受你启动 shell 的 `readonly` 保护。

也就是说，`readonly` 只能限制当前 bash 里后续再赋值：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1
```

它不能阻止 vLLM/Ray/Python worker 在自己的进程里改 `os.environ`。

另外，如果 `readonly ASCEND_RT_VISIBLE_DEVICES` 没有 `export`，它本身也不一定会作为环境变量传给子进程。你的 Python 进程是否能看到它，取决于最终进程环境，而不是 shell readonly 属性。

## 6. 对你这个问题的直接回答

### 1. `ASCEND_RT_VISIBLE_DEVICES` 到底是谁改的？

按代码能证明的情况分两类：

如果你使用的是普通 multiproc/uniproc：

- `vllm-ascend/worker/worker.py` 没有改它
- `torch.npu.set_device()` 发生在你的日志之后，不可能解释日志前的变化
- vLLM multiproc/uniproc 路径没有看到在 `init_device()` 前写 `ASCEND_RT_VISIBLE_DEVICES`
- 那么它只能来自当前 worker 进程继承到的环境，或者进入 vLLM 前的外部环境/启动器

如果你使用的是 RayExecutorV2：

- 明确可能是 `RayExecutorV2` 从 driver 进程环境通过 `driver_env_vars` 传给 Ray worker 的
- 设置点是：

`E:/lizy/code/vllm-project/vllm/vllm/v1/executor/ray_executor_v2.py:152`

```python
os.environ.setdefault(key, value)
```

其中 `ASCEND_RT_VISIBLE_DEVICES` 没有被排除，所以会被传递。

如果你使用的是旧 RayExecutor，并且配置了额外复制 env：

- 设置点是：

`E:/lizy/code/vllm-project/vllm/vllm/utils/system_utils.py:34`

```python
os.environ[k] = v
```

调用来自：

`E:/lizy/code/vllm-project/vllm/vllm/v1/worker/worker_base.py:222`

```python
update_environment_variables(envs)
```

### 2. 为什么启动脚本没设置它，但 Python 进程里能看到变化？

代码层面有三个可证明原因：

第一，启动脚本没设置，不代表上层环境没有。

比如启动这个脚本的外层 shell、systemd、Ray worker runtime、容器环境、远程执行器，都可能已经有该变量。你的脚本 `unset` 只影响当前 shell 及其直接派生流程，不影响已经存在的 Ray worker 环境或其他启动器环境。

第二，RayExecutorV2 会传播 driver 环境。

只要 driver 进程里存在 `ASCEND_RT_VISIBLE_DEVICES`，RayExecutorV2 的：

`E:/lizy/code/vllm-project/vllm/vllm/v1/executor/ray_env_utils.py:18`

```python
return {key: value for key, value in os.environ.items() if key not in exclude_vars}
```

会把它收集进去，因为排除列表里没有 `ASCEND_RT_VISIBLE_DEVICES`。

然后：

`E:/lizy/code/vllm-project/vllm/vllm/v1/executor/ray_executor_v2.py:152`

```python
os.environ.setdefault(key, value)
```

会在 worker 初始化前设置它。

第三，bash 的 `readonly` 不能保护 Python 进程内的 `os.environ`。

所以即使你写了：

```bash
readonly ASCEND_RT_VISIBLE_DEVICES
```

也不能阻止 worker 进程内部设置自己的环境变量。

### 3. 变化可能发生在 vLLM、vllm-ascend、Ascend runtime、Ray/多进程 worker、还是 shell 环境？

按代码证据排序：

1. **RayExecutorV2：最明确的代码路径**
   - 能在 `NPUWorker.init_device()` 前把 driver env 传给 worker
   - `ASCEND_RT_VISIBLE_DEVICES` 没有被排除
   - 代码位置：`ray_executor_v2.py:152`

2. **旧 RayExecutor + env copy：可能**
   - 默认不复制 `ASCEND_`
   - 但如果配置了 `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=ASCEND_RT_VISIBLE_DEVICES`，会复制并覆盖
   - 代码位置：`ray_executor.py:309`、`worker_base.py:222`、`system_utils.py:34`

3. **普通 multiproc：未找到写入点**
   - worker 进程继承环境
   - vLLM 只传递 `assigned_physical_gpu_ids` 映射，不直接写 `ASCEND_RT_VISIBLE_DEVICES`

4. **vllm-ascend：未找到生产路径在 `init_device()` 前写它**
   - 主要是声明和读取

5. **Ascend runtime / `torch.npu.set_device()`：不能解释你这条日志**
   - 因为你的日志在 `torch.npu.set_device()` 之前

6. **shell 的 `readonly`：不能阻止 Python 进程改自己的 env**
   - 所以不能作为“不会变化”的保证

## 7. 对你当前启动脚本的判断

你启动命令里没有显式指定 Ray backend：

```bash
vllm serve ... \
    --data-parallel-size 2 \
    --tensor-parallel-size 8
```

如果实际 executor 是本地 multiproc，那么代码链路上没有看到 vLLM/vllm-ascend 在 `NPUWorker.init_device()` 前设置 `ASCEND_RT_VISIBLE_DEVICES`。

这种情况下，如果日志里出现了非空值，最该查的是：

1. `vllm serve` 进程启动前，driver 进程真实环境里是否已经有它
2. 是否通过 Ray backend 启动了 worker
3. 是否使用了 RayExecutorV2
4. 是否设置过 `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`
5. 是否有外层 launcher、容器、systemd、mpirun、ray runtime_env 注入了它

如果你看到的是 RayExecutorV2 路径，那么最可能的代码原因就是：

```text
driver os.environ 里已有 ASCEND_RT_VISIBLE_DEVICES
  -> RayExecutorV2.get_driver_env_vars() 收集它
  -> RayWorkerProc.initialize_worker() setdefault 写入 worker os.environ
  -> NPUWorker.init_device() 打印出来
```

对应代码：

- `vllm/vllm/v1/executor/ray_env_utils.py:18`
- `vllm/vllm/v1/executor/ray_utils.py:33`
- `vllm/vllm/v1/executor/ray_executor_v2.py:152`
- `vllm/vllm/v1/executor/multiproc_executor.py:625`
- `vllm-ascend/vllm_ascend/worker/worker.py:458`

## 最短结论

从这两个仓库代码看：

- `vllm-ascend` 没有在 `NPUWorker.init_device()` 前主动改 `ASCEND_RT_VISIBLE_DEVICES`
- `torch.npu.set_device()` 也不是原因，因为日志在它之前
- 普通 multiproc/uniproc 路径没有发现写入点
- RayExecutorV2 路径明确可能在 `init_device()` 前把 driver 环境里的 `ASCEND_RT_VISIBLE_DEVICES` 传播到 worker
- 你的 `readonly ASCEND_RT_VISIBLE_DEVICES` 只约束当前 shell，不约束 Python/Ray worker 进程内的 `os.environ`

所以，要定位你现场“到底是谁改的”，下一步最有效是在这三个位置分别打日志：

1. `vllm/vllm/v1/executor/ray_executor_v2.py:331` 后面打印 `self.driver_env_vars.get("ASCEND_RT_VISIBLE_DEVICES")`
2. `vllm/vllm/v1/executor/ray_executor_v2.py:152` 前后打印 worker 里的 `os.environ.get("ASCEND_RT_VISIBLE_DEVICES")`
3. `vllm-ascend/vllm_ascend/worker/worker.py:458` 继续保留你现在的日志

如果第 1 个位置已经有值，就证明来源是 driver 环境；如果第 1 个没有、第 2 个 worker 已经有值，就说明是 Ray actor/runtime 环境提前带入。
