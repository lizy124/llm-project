# 04 环境与操作 — 阶段性归档

## 身份与起点

P=80.5.9.135（node-97-36），D=80.5.9.133（vllm-dev-a3-133），各16个A3芯片；两机已有lizy_1009容器，使用day20261012镜像，完整标识sha256:723f75865d785bcafdc0bd7be6999695ffa5777d4cfd24486cfbd0aa742e0b0e。既有镜像构建过程未在本任务复现；docker_inspect.json保存挂载/配置。本归档以这两个既有容器及共享模型/数据为复现起点。

模型/mnt/share/weights/DeepSeek-V4-Flash-0731-w8a8，API模型名model。共享/mnt/share；/home为每机本地目录，需分别准备scripts。容器Python3.11.10、torch2.10.0、torch-npu2.10.0.post7、numpy1.26.4、memcache-hybrid与memfabric-hybrid均1.3.0；135 pyarrow25.0.1。精确版本及CANN记录见evidence/20261013_pinned_revalidation_214027_retry1/inputs/{135,133}/。

135可免密ssh；135→133已免密。客户端直连133须使用D:/project/kv_pool/ssh_known_hosts_133_probe隔离known_hosts。本次复验经135转至133，未改用户全局known_hosts。

## 准备与资源检查

进入对应容器，cd /home/lizhongyang/1011/scripts；将03的启动脚本和mmc配置放在同目录，将测试源放在aisbench、定向工具放在/home/lizhongyang/1011/revalidation_tools。原代理Python需位于proxy.sh引用的外部路径。保留原冻结数据，不运行会删除数据的ab_auto.sh。

每轮开始前停止本套P/D，池化额外停止Meta/standalone；只操作目标容器内已确认服务。npu-smi info核对16芯片空闲与健康，ss核对8910/8920/8100/5000/6000/8004/12321端口。共享环境被他人使用时不得启动复验；本次用户因此要求中止。

AISBench会替换安装目录vllm_api_chat_temp.py、gsm8k/test.jsonl，并可能创建train.jsonl，测前用cp -a保存原状态及temp_api.py内容。恢复时按内容/链接处理，核对哈希/链接目标，不强制复制挂载路径不支持的属性。版本采证在中立cwd执行，避免包内platform.py遮蔽标准库。

## 启动

每条均在对应容器的scripts目录，前台服务各独立终端。池化先切pr-18277，然后135 bash meta.sh，两机各bash standalone.sh；确认Successfully initialized memcache后，135 bash p.sh、133 bash d.sh；两端health成功后135 bash proxy.sh。

无池化先切master，135 bash p_nopool.sh、133 bash d_nopool.sh，health成功后135 bash proxy.sh。后台docker exec使用bash -c，不能用bash -lc。135:8910与133:8920的/health返回200、proxy8100监听才可测试；不额外发送Hello或随机请求以免改变缓存。

## 测试与证据

在135容器执行：DTS_PIN_PREFIX_RESULT=/mnt/share/lizhongyang/dts_zyd_ab/<run_id>/<round>/raw/pinned_warmup.json bash /home/lizhongyang/1011/revalidation_tools/run_pinned.sh。完整参数与代码见03。每轮核对冻结正式SHA256932522b95ec848cfbf6718e480b2925768d498d008ae20e719ab1e1121958614、前缀ad4bf6bbe17b5bf3423858045bcbb4267eae963fb37b4bbc282cbe8782f2b59b，以及每DP预热查询和196/196成功。不开profiler。

实际编排入口135宿主机bash /tmp/dts_zyd_revalidate_ab.sh <run_id>；所需host/stop/analyze辅助工具按03源码分别传入135与133。实际来源脚本、组件版本、commit、哈希、日志和真实请求响应均在evidence中。P/D日志时钟有偏差，按轮次对齐，吞吐均取同一135客户端正式CSV最后一行。

## 停止与恢复

本次已向记录且身份核对过的编排PID发送TERM，退出陷阱收集当前轮日志、清理两机目标容器的测试进程、切master、恢复AISBench。额外检查两机最终ps不含测试服务、git工作区干净，分支master。停止原因是用户要求让出共享环境，退出130不是性能测试失败。原始停止/恢复证据见05。不改其他容器，不把其他使用者的NPU占用判为本套服务残留。
