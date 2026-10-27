# Forecaster 训练、预测与回归验证

只为实验选中的区域训练独立 LSTM/XGBoost，权重保存在数据集的
`cache/weight/<backend>/`。先检查训练资格，再执行显式选区或自动抽样；
训练不依赖预测起点、alpha、Agent 模式和实验种子。

`history_days` 决定输入 H 小时，`horizon_days` 决定直接输出 P 小时，默认 720→24。
连续 N 个小时以步长 1 产生 N-H-P+1 个完整样本；输入和标签都不得跨缺口。
样本按时间顺序以 80%/20% 划分，删除标签触及验证起点的训练样本。
两个比例必须均大于零且总和为 1；CLI 可一起设置为
`--training-train-ratio 0.7 --training-validation-ratio 0.3`。
预处理仅拟合训练前缀。模型单次训练、验证早停并保留最佳权重，不再全量重训。
LSTM 一次输出 P 个值；XGBoost 保存 P 个独立回归器，分别选择最佳树数。
默认至少 45 天有效小时和 512 个完整样本，720→24 窗口需至少 1,255 个连续小时。
训练报告记录样本数、删除数、实际维度、划分比例、最佳轮数和预处理范围。

默认训练参数固定，不再按样本量自动切换档位：LSTM 为 hidden32、1 层、batch64、
学习率 0.001、最多 100 epochs；XGBoost 为 depth3、每个输出最多 800 棵树、学习率 0.03。
`training.defaults`、`dataset_overrides`、`zone_overrides` 管理覆盖，
优先级为 CLI > 区域 > 数据集 > 全局 > 内置默认。XGBoost 使用树数与学习率，
没有 epoch 或 batch 设置。旧 `run` 下训练键、`lstm_context_hours`、`xgboost_lags`、
`training.validation_days` 和 `min_reference_samples` 均报错。

两种后端默认 CUDA，可显式指定 CPU。LSTM 多 worker 使用独立进程，XGBoost 使用线程；
worker 数必须为正整数，没有跨数据集的全局训练上限。相同模型键仍有缓存写入锁。
输入/输出长度、区域内容和实际训练参数参与模型身份，其他区域改变不会使当前区域失效。
缓存版本不可变；预测产物绑定具体 model_id，Agent-only 仅加载固定版本。

五种后端共用默认 30 天历史、5 天校准、1 天预测：对于起点 T，历史为
[T-35天,T-5天)，校准为 [T-5天,T)，预测为 [T,T+1天)。
每个校准日只使用当日起点前已知历史。LSTM/XGBoost 每次计算完整 P 小时输出，
取前 24 小时校准；其他后端每次验证 24 小时。原始预测与冻结日内均值混合一次，
按五日相同预测偏移平均“混合预测−真实负荷”，得到 24 个 bias。
正式输出混合后减 bias 并裁剪非负，多日循环使用这 24 个 bias。
`validation_days: 0` 使用零 bias；缺失验证小时或非法负荷明确报错。
调价重预测复用模型和 bias，不重新训练或校准。

配置只使用 `forecast_starts`、`forecast_models` 和复数 alpha 键；单次 Python 接口及
产物使用 `forecast_origin`、`backend`。模型独立 alpha 覆盖全局列表，实验按有效列表展开。
输出路径包含实际 `blend_<alpha>/<backend>`；六位小数精度同时用于参数与路径。
模型权重可能使用历史预测区间内或之后的数据进行拟合/早停。结果记录训练范围、
逐小时重叠和 model_id；只有预测起点不早于训练及早停使用数据的结束时间，才标为
`evaluation_is_out_of_sample`。评估按 model_set_id 区分版本。旧预测器协议不回放，
旧结果字段兼容仅发生在离线读取边界。

| 文件 | 职责 |
|---|---|
| `training_config.py`、`config.py`、`main.py` | 配置验证、覆盖和训练/预测阶段入口 |
| `zone_training.py` | 完整窗口计数、资格、选中区域训练、模型版本和校验 |
| `offline_models.py` | 直接多步模型、单次早停、预处理及加载预测 |
| `forecasting.py`、`global_forecaster.py` | 共用校准、原生产物和调价复用 |
| `orchestrator.py` | 选区后训练、实验矩阵与阶段交接 |
| `reporting.py`、`evaluation_artifacts.py` | 结果与权重身份、历史字段读取兼容 |

运行全量回归：

```powershell
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

关键覆盖包括完整 H+P 窗口和标签隔离、选中区域训练、窗口长度缓存失效、单次早停和
训练前缀预处理、真实 CPU/CUDA LSTM/XGBoost、跨数据集并发及独立进程、五日 bias、
多日完整输出、无未来负荷泄漏、JSON/权重校验、不同 alpha 共用原始预测，以及
Forecaster→Agent-only 复用。默认一天对应 8 个三小时控制窗口。
TimesFM/Chronos 流程测试使用确定性后端；Agent 端到端测试使用离线模式。
