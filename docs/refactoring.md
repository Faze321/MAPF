# Forecaster 公共流程与回归验证

TimesFM、Chronos、LSTM、XGBoost 和 AR 共用训练、逐日验证、日内混合和 bias 校正流程。
默认使用 30 天训练、5 天验证、1 天预测；三个长度可配置，模型上下文长度单独设置。
训练权重、标准化参数、AR 价格响应和日内均值曲线只从训练区间取得。

验证每次预测 24 小时，只读取该日起点前已知的真实历史，不重新训练。
模型输出与冻结的训练日内均值混合一次；五日同一预测偏移小时的
`混合预测 - 真实负荷` 取平均，得到 24 个 bias。
正式预测执行相同混合，再减去 bias 并裁剪为非负值。
多日预测循环使用这 24 个 bias，`validation_days: 0` 则使用零 bias。
验证窗口必须完整，负荷必须有效；LSTM 和 XGBoost 在预测期只递推自己的预测。

公共逻辑集中在以下位置：

| 位置 | 职责 |
|---|---|
| `config.py`、`main.py` | 解析复数日期、模型和 alpha 配置；CLI 覆盖同名 YAML 后应用独立 alpha 优先规则 |
| `forecasting.py` | 五后端共用时间边界、一次训练、逐日校准、点预测及区间校正 |
| `xgboost_forecaster.py` | XGBoost 训练、原始递推和模型 JSON 持久化 |
| `global_forecaster.py` | 保存与加载原生预测器产物；调价重预测复用模型及同一 bias |
| `orchestrator.py` | 按每个后端的有效 alpha 列表展开实验，统一各阶段路径和产物身份 |
| `dataset_adapter.py`、`data_loader.py`、`zone_eligibility.py` | 数据适配、预测起点之前的特征、时间划分及区域资格缓存 |
| `reporting.py`、`prompts.py` | 写入当前字段，构建不包含预测期真实负荷和评估信息的 Agent 上下文 |
| `evaluation_artifacts.py` | 在离线读取边界统一历史结果字段；新评估输出只写当前名称 |
| `usage.py` | Agent 调用、轮次和累计 Token 统计，保留缺失用量的不完整标记 |

配置只使用 `forecast_starts`、`forecast_models` 和复数 alpha 键；单次实验也使用单元素列表。
Python 单次调用和产物使用 `forecast_origin`、`backend`，实际 alpha 保持为一个数值。
旧单数配置、CLI、`roll_actuals` 和模型级 `step_horizon` 已移除。
输出目录始终包含 `blend_<alpha>/<backend>`，alpha 以六位小数精度去重和标识。
预测身份只包含当前模型参数和实际 alpha，不包含其他候选配置。

原生模型产物使用 schema 3，Forecaster 清单和控制结果使用 schema 4，
离线评估报告使用 schema 2。旧预测器协议不迁移、不回放；旧结果仅供离线评估读取。
Agent 调价阶段不重新训练、不重新估计 bias。

运行全量回归：

```powershell
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

关键测试包括：

- `test_forecast_calibration.py`：720/120/24 小时时间边界、自定义窗口、五日逐小时均值、
  混合和误差符号、无泄漏、状态复用，以及真实小型 CPU LSTM。
- `test_forecast_config_v2.py`：全局继承和独立覆盖、单值和多值、非法配置、实验次数、路径隔离。
- `test_forecast_output_v2.py`：产物身份、schema、当前输出字段与离线历史字段兼容。
- `test_forecast_intervals.py`：无效负荷拒绝，以及分位数缺失时使用全部验证日残差生成区间。
- `test_xgboost.py`：真实 XGBoost、JSON 往返、调价重预测，以及
  Forecaster → Agent-only 复用；默认一天输出对应 8 个三小时控制窗口。
- 数据集、提示词、缓存、控制和评估测试继续检查跨数据集隔离、无泄漏上下文及结果一致性。

TimesFM 和 Chronos 的流程测试使用确定性模拟后端，不下载大模型；
Agent 端到端测试使用离线模式，不调用真实 Agent API。
