# 公共逻辑与回归验证

本次重构保留 `main.py` 入口、配置字段、命令行优先级、数据集目录隔离和输出结构。
模型算法、Agent 提示词、三次控制尝试及负荷分级阈值保持原样。

公共逻辑集中在以下位置：

| 位置 | 职责 |
|---|---|
| `config.py` | 从 `RunConfig` 取得数值默认值；共用列表解析；统一向下传递的模型参数名称 |
| `usage.py` | Agent 调用、轮次和累计 Token 统计；缺失用量保持不完整标记 |
| `dataset_adapter.py` | 标准长表转宽表、原子写入，以及缓存和输出共同使用的进程锁 |
| `forecasting.py` | TimesFM、Chronos、LSTM 共用预测分段；一次预测只建立一次时间索引 |
| `orchestrator.py` | 用统一表映射收集和保存实验汇总，沿用各文件名与字段 |
| `global_forecaster.py` | 按站点累计预测偏移，避免重复扫描已经生成的预测记录 |

原有 Token 工具函数仍可从 `agents.py`、`orchestrator.py`、`reporting.py` 导入。
缓存锁仍等待写入完成，输出锁仍在冲突时拒绝启动；两者共用底层实现。
扩展模型参数时，需同步检查公共参数名称、后端接口和持久化模型格式。

运行回归测试：

```powershell
python -B -m unittest discover -s tests -v
```

本次验证包含 35 项测试，以及重构前后的真实数据对照：

- UrbanEV、CHARGED/LOA、MP-EVData，使用 AR 跑完五种离线 Agent 模式。
- 426 个 JSON/CSV 文件在统一输出根目录、创建时间和完成时间后内容一致。
- 三数据集的 LSTM 单轮并行训练完成；42 个 CSV 与原结果一致，数值容差为 `rtol=1e-7`、`atol=1e-8`。
- TimesFM、Chronos、LSTM 的分段预测和模型复用通过确定性模拟后端测试；本次没有重新下载大模型或调用真实 Agent API。

本机详细对照记录位于 `output/refactor_validation/comparison.json` 和
`output/refactor_validation/refactor_metrics.json`，这两个文件为不纳入版本控制的验证产物。
