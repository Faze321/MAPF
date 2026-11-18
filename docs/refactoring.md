# Forecaster 训练、预测与回归验证

只为实验选中的区域训练独立 LSTM/XGBoost 和 TimesFM/Chronos LoRA，权重保存在各数据集缓存的
`weight/<backend>/`。先检查训练资格，再执行显式选区或自动抽样；
训练不依赖预测起点、alpha、Agent 模式和实验种子。

默认缓存位于 `<project>/cache/<数据集相对于项目 data/ 的路径>/`，例如
`cache/UrbanEV`、`cache/MP-EVData` 和 `cache/CHARGED/JHB`。
项目 `data/` 之外的数据源使用 `cache/_external/<safe-leaf>_<path-hash>/`，
散列为规范化绝对数据路径的 SHA-256 前 12 位十六进制字符。
显式 `data.cache_dir` 或 `--cache-dir` 精确指定该数据集的缓存目录。
旧缓存和模型文件不自动移动，新默认运行在新位置构建缓存；
已有预测产物仍引用旧权重，需保留原文件以便重放。

`history_days` 决定输入 H 小时，`horizon_days` 决定直接输出 P 小时，默认 840→24。
连续 N 个小时以步长 1 产生 N-H-P+1 个完整样本；输入和标签都不得跨缺口。
样本按时间顺序以 80%/20% 划分，删除标签触及验证起点的训练样本。
两个比例必须均大于零且总和为 1；CLI 可一起设置为
`--training-train-ratio 0.7 --training-validation-ratio 0.3`。
预处理仅拟合训练前缀。模型单次训练、验证早停并保留最佳权重，不再全量重训。
LSTM 一次输出 P 个值；XGBoost 保存 P 个独立回归器，分别选择最佳树数。
默认至少 45 天有效小时和 512 个完整样本，840→24 窗口需至少 1,375 个连续小时。
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

`timesfm`、`timesfm_lora`、`chronos`、`chronos_lora`、`lstm`、`xgboost`、`AR`
是独立后端。原版 TimesFM/Chronos 始终使用预训练权重，选择 `_lora` 变体才会训练
并加载各站点适配器；可在同一矩阵中比较原版与 LoRA。变体共用原版的 repo 配置，
输入长度按 `history_days`，原版按各自 `*_context_hours`。LoRA 超参数仍放在
`training.defaults` 下的 `timesfm_*`、`chronos_*` 键中，覆盖优先级不变。
旧 `*_lora_enabled` 开关明确报错。适配器分别写入数据集缓存的
`weight/timesfm_lora/`、`weight/chronos_lora/`，不迁移旧 LoRA 产物。

七种后端默认使用 [T-35天,T) 的历史预测 [T,T+1天)。
`run.validation_bias` 默认关闭。`run.validation_bias_by_dataset` 按数据集目录覆盖，
显式 CLI `--validation-bias` 或 `--no-validation-bias` 优先。
开启时，`run.validation_days` 指定预测前用于校准的完整天数，取正整数，默认 7，
可用 `--validation-days` 覆盖。在 35 天历史和 7 天校准的设置下，
历史参考窗口为 [T-42天,T-7天)，校准窗口为 [T-7天,T)。
每个校准日只使用当日起点前已知历史。LSTM、XGBoost、`timesfm_lora` 和
`chronos_lora` 每次计算完整 P 小时输出，取前 24 小时校准；
原版 TimesFM、Chronos 和 AR 每个验证日直接预测 24 小时。原始预测与冻结日内均值混合一次，
对所有校准日同一小时的“混合预测−真实负荷”取平均，得到 24 个 bias。
正式输出混合后减 bias 并裁剪非负，多日循环使用这 24 个 bias。
开启校正时，缺失验证小时或非法负荷会报错。关闭时直接使用日内混合结果。
调价重预测复用模型和 bias，不重新训练或校准。

配置只使用 `forecast_starts`、`forecast_models` 和复数 alpha 键；单次 Python 接口及
产物使用 `forecast_origin`、`backend`。模型独立 alpha 覆盖全局列表，实验按有效列表展开。
LoRA 的独立列表为 `timesfm_lora_diurnal_blend_alphas` 和
`chronos_lora_diurnal_blend_alphas`，不继承原版模型的独立设置；`null` 继承全局。
输出路径包含实际 `blend_<alpha>/<backend>`；六位小数精度同时用于参数与路径。
模型权重可能使用历史预测区间内或之后的数据进行拟合/早停。结果记录训练范围、
逐小时重叠和 model_id；只有预测起点不早于训练及早停使用数据的结束时间，才标为
`evaluation_is_out_of_sample`。评估按 model_set_id 和实际校准设置分组。
历史结果字段由离线评估在读取时识别。

当前 Native 预测器产物为 schema 8，Forecaster manifest 为 schema 10。
Agent-only 可读取 manifest 9 的七日校正或未校正产物，以及未校正的 native 7、manifest 8。
更早的产物需先通过 `forecaster` 或 `full` 阶段重新生成。
LSTM/XGBoost 离线权重仍为 schema 2，缓存位置、训练内容及参数匹配时可复用，
无需因此强制重新训练；旧 LoRA 权重不自动迁移到新后端身份。

| 文件 | 职责 |
|---|---|
| `training_config.py` | 原版/LoRA 后端身份、共享模型家族映射及离线训练配置 |
| `config.py`、`main.py` | 运行配置验证、覆盖和训练/预测阶段入口 |
| `foundation_models.py` | 基础模型 LoRA 训练、适配器加载和已训练权重预测 |
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
训练前缀预处理、真实 CPU/CUDA LSTM/XGBoost、跨数据集并发及独立进程、按数据集启用 bias 并配置校准天数、
多日完整输出、无未来负荷泄漏、JSON/权重校验、不同 alpha 共用原始预测，以及
Forecaster→Agent-only 复用。默认一天对应 8 个三小时控制窗口。
TimesFM/Chronos 流程测试使用确定性后端；Agent 端到端测试使用离线模式。
