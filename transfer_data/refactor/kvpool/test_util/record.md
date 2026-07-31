# PR 13160 验证前期记录

目标 PR：`https://github.com/vllm-project/vllm-ascend/pull/13160`

## 已完成

- 已确认 `vllm-ascend` 为 editable 安装。
  - 源码路径：`/vllm-workspace/vllm-ascend`
  - 包入口：`/vllm-workspace/vllm-ascend/vllm_ascend/__init__.py`
  - 版本：`0.19.1rc2.dev1239+g2b1ac6124`
- 已确认 `vllm` 也为 editable 安装。
  - 源码路径：`/vllm-workspace/vllm`
  - 包入口：`/vllm-workspace/vllm/vllm/__init__.py`
  - 版本：`0.25.1+empty`
- 已确认 `/vllm-workspace/vllm-ascend` 工作区干净后，拉取并切换到 PR 分支。
  - 分支：`pr-13160`
  - 当前提交：`3a0404d2d`
- 已检查 `/vllm-workspace/vllm` 当前状态。
  - 状态：detached HEAD
  - 当前提交：`752a3a5`
- 当前环境没有 `gh` 命令，暂未直接读取 PR 元数据。

## 待确认

- PR 13160 是否依赖配套的 `vllm` 主仓分支或特定 commit。
- 如果后续导入检查、UT 或服务启动出现 `vllm` API 不匹配，再根据 PR 描述/CI 配置切换 `/vllm-workspace/vllm` 到对应版本。

## 建议下一步

- 先在当前 `vllm=752a3a5` + `vllm-ascend=pr-13160` 组合下做语法/导入检查。
- 再跑 AscendStore 相关 UT。
- 最后按 `test_pool_overview.md` 优先验证 Mooncake PD-Mixed 单实例链路。
