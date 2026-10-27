# 多数据集训练

项目支持 UrbanEV、CHARGED 和 MP-EVData。数据读取统一在 `dataset_adapter.py`，
并行调度统一在 `orchestrator.py`，训练入口统一为 `python main.py`。

## 下载和来源

在项目根目录运行（使用项目 `.venv` 或 `uv run`）：

```powershell
python download_datasets.py charged mp_evdata
# 也可只下载某一个 CHARGED 城市
python download_datasets.py charged --cities JHB
```

下载器获取 [CHARGED 官方仓库](https://github.com/IntelligentSystemsLab/CHARGED)
的当前 commit，再从该固定 commit 下载六城市 `remove_zero` 版本的小时级
volume、energy/service price、weather、sites、POI 和 info 文件。保存时校验 Git blob
SHA-1。MP-EVData 来自 [Figshare 29882366](https://doi.org/10.6084/m9.figshare.29882366)，
获取带不可变 file ID 的小时负荷、电价和 README，校验发布方 MD5。
每次成功下载保存 `download_manifest.json`。已有文件校验一致时跳过下载。
不下载、不执行上游训练脚本，不生成合成数据；不需要下载原始会话或 15 分钟文件。

MP-EVData 的发布许可为 CC BY 4.0。CHARGED 的数据和辅助来源使用条件请参照其
官方发布说明；引用数据论文：[CHARGED](https://www.nature.com/articles/s41597-025-05584-7)、
[MP-EVData](https://www.nature.com/articles/s41597-026-07273-5)。

## 修改配置切换数据集

在 `config.yaml` 的现有 `data`、`run` 段中修改这些字段，然后运行 `python main.py`：

```yaml
data:
  adapter: auto
run:
  data_dir: data/MP-EVData
  forecast_starts: null
  forecast_models: [timesfm]
  history_days: 30
  validation_days: 5
  horizon_days: 1
  zones: null
```

只需更改 `run.data_dir` 即可切换：

| 数据集 | 文件夹 |
|---|---|
| UrbanEV | `data/UrbanEV` |
| CHARGED | `data/CHARGED/AMS`、`JHB`、`LOA`、`MEL`、`SPO`、`SZH` 对应的城市目录 |
| MP-EVData | `data/MP-EVData` |

也可以填写其他绝对路径或相对于当前工作目录的路径。识别依据是原始文件布局，
文件夹可以改名。CHARGED 必须指向包含 `volume.csv`、`e_price.csv`、`sites.csv` 的
城市文件夹；UrbanEV 使用 `volume.csv`、`e_price.csv`、`inf.csv`；MP-EVData 使用
`station-level load Profile 1h.xlsx` 和 `price.xlsx`。缺失或混合格式会明确报错。
自定义长表继续使用 `data.adapter: long_format` 和原有字段映射。

`forecast_starts` 留空且 `forecast_start_count: 1`（默认）时，
使用当前数据最后 `horizon_days` 天作为预测窗口。默认预测起点为 `T` 时，
预测历史取 `[T-35天, T-5天)` 的 720 小时，Bias 验证取 `[T-5天, T)` 的 120 小时，
正式预测 `[T, T+1天)` 的 24 小时。三个天数可以独立配置。
每天验证只读取当天起点之前的真实历史，五日同一预测偏移小时的残差均值形成
24 个 bias；调价重预测复用已保存模型及 bias。LSTM/XGBoost 权重来自独立离线训练，
`history_days` 同时决定离线训练样本和预测的输入长度，`horizon_days` 决定直接输出长度。
默认每个样本输入 720 小时、输出 24 小时；预测历史还用于固定日内均值曲线。
`zones: null` 从当前数据自动选站。
显式填写日期或站点时会按你的配置执行，因此跨数据集切换时需使用该数据集的日期和编号。
预测模型、Agent 模式等实验参数使用 `run`；LSTM/XGBoost 的训练参数使用独立
`training` 配置，旧的 `run.lstm_*`、`run.xgboost_*` 训练键会明确报错。
各模型复数 alpha 配置仍属于 `run`。LSTM 与 XGBoost 默认使用 CUDA 进行离线训练和推理。
`training.defaults.lstm_device` 和 `training.defaults.xgboost_device` 均设为 `cuda`，
可用 `cuda:0` 指定 GPU，或显式设为 `cpu`；CLI 对应 `--lstm-device`、`--xgboost-device`。
XGBoost 继续使用 `tree_method: hist`。设备设置参与训练指纹，修改设备会选择或创建对应的模型版本。
日期和模型统一使用 `forecast_starts: [日期]`、`forecast_models: [模型]`，
CLI 对应 `--forecast-starts`、`--forecast-models`，单次实验也使用单元素列表。
旧的单数日期、模型和 alpha 配置会明确报错。Python 单次调用与输出字段使用
`forecast_origin` 和 `backend`。

每个数据集随机选择 5 个区域，可设置：

```yaml
run:
  zones: null
  experiment_zone_count: 5
  experiment_zone_selection: random
  experiment_zone_seed: 42
```

若每个数据集需要不同的区域数量，保留默认值并增加按目录配置的覆盖项：

```yaml
run:
  experiment_zone_count: 5
  experiment_zone_counts:
    data/UrbanEV: 10
    data/CHARGED/JHB: 3
    data/MP-EVData: 5
```

`experiment_zone_counts` 的键对应 `run.data_dir` 中的数据集文件夹，值必须为正整数。
未单独填写的目录使用原有 `experiment_zone_count`；省略该映射或设置为 `{}`/`null`
即可继续统一设置数量。与 `data_dir` 一样，相对路径以程序运行目录为基准，
等价的相对路径和绝对路径会匹配同一目录；同名但位于不同位置的文件夹可以分别配置。
单数据集和并行子进程都采用此规则，`--data-dir` 指定目录时也会自动匹配对应数量。
数量优先级为 `--experiment-zone-count` > 对应目录覆盖值 > 默认数量；
显式 `zones`/`--zones` 仍优先指定区域编号，不会被这些数量设置截断。
覆盖项同时适用于随机选区和代表性选区。

程序先确定预测起点，再从缓存的合格区域列表中抽样一次；同一数据集的全部起点、
模型、Agent 模式和实验种子共用这组区域。更改 `experiment_zone_seed` 可重新抽样，
区域不重复。随机选区时，符合条件的区域不足该数据集的有效配置数量时会明确报错并给出资格缓存位置，
不会补入恒定区域，也不会静默减少实验区域数。

随机抽样同时要求负荷和 `energy_price` 非恒定。判定范围为该区域从数据开始至
**最早预测起点减去 `validation_days`** 的全部历史，右端不包含，不使用验证或预测期数据。
负荷或电价任意一项恒定都会被排除；常数比较使用 `rtol=1e-9`、`atol=1e-12`，
容忍浮点计算误差。每个预测起点的 `history_days` 天上下文窗口还必须有完整、有效的小时数据。
这是对早期历史中区域是否恒定的判断，不要求每个上下文窗口都发生变化；
例如最近一周电价固定，但更早历史有变化的区域仍可进入候选。
真实零负荷保留，不新增零值比例或稀疏度阈值。

显式 `zones` 或 `--zones` 优先于自动抽样，不强制套用上述随机选区规则。
默认 `representative` 保持原有代表性评分选区方式。

每个数据集自动选择两个均匀分布的预测起点，可增加：

```yaml
run:
  forecast_starts: null
  forecast_start_count: 2
```

程序按天筛选有效起点：起点之前必须包含 `history_days + validation_days` 天的
连续小时记录，之后必须包含完整的 `horizon_days` 天预测序列，再取候选日期中约 1/3 和 2/3
位置的两个起点（均为 00:00）。这里的天数均按每天 24 小时计算。
随机选区时，先按适配器转换后全部可用区域的共同覆盖范围筛选日期，再读取对应日期的
选区资格缓存并随机抽样，因此更改抽样种子不会改变自动起点。两次实验共用同一组区域。
显式选区按选中区域的共同覆盖范围筛选；代表性选区保持先按全部可用区域的共同覆盖
筛选日期，再使用第一个起点之前的数据选区。存在缺失小时的窗口不会作为候选，
候选不足会明确报错。
`forecast_start_count` 可设为更大的正整数；手动指定 `forecast_starts`
或 `--forecast-starts` 时，手动日期优先，不再自动取点。

原先新增的 `--dataset`、`--city`、`--device` 和 `datasets` 配置块已移除，
不再需要 `train_datasets.py`。原有模型、阶段等 CLI 参数仍可覆盖配置；
`--data-dir` 可临时指定单个文件夹。

## 各区域独立离线训练

先筛选训练资格并完成实验选区，只为选中区域训练独立的 LSTM 和 XGBoost：

```powershell
python main.py --stage train --forecast-models lstm xgboost
```

`train`、`forecaster`、`full` 使用相同的显式站点或自动抽样规则；未选中区域不会训练。
每个区域只使用自己的有效历史，训练不依赖预测起点、alpha、Agent 模式或实验种子。
默认至少需要 45 天有效小时记录和 512 个完整输入/输出样本。
设输入 H 小时、输出 P 小时，N 个连续小时按步长 1 产生 N-H-P+1 个样本；
默认 H=720、P=24 时至少需要 1,255 个连续小时才能满足样本数门槛。
负荷或电价恒定、重复时间戳和样本不足会明确报错。无效小时形成缺口，
样本的输入与标签都不能跨缺口；不插值补负荷。

完整样本按时间顺序以 `training.train_ratio: 0.8`、`validation_ratio: 0.2` 划分。
两个比例都必须大于零且总和为 1；CLI 可一起设置为
`--training-train-ratio 0.7 --training-validation-ratio 0.3`。
删除输出标签触及验证起点的训练样本，记录实际训练/验证数量和删除数量。
预处理只拟合训练前缀。模型仅训练一次，LSTM 按验证误差早停并恢复最佳权重；
XGBoost 为 P 个输出小时分别训练和保存最佳树数。不会重新拟合验证数据。
该内部验证与预测前的 5 天 Bias 校准相互独立。

固定默认值为 LSTM 隐藏大小 32、1 层、batch 64、学习率 0.001、最多 100 epochs；
XGBoost 最大深度 3、每小时模型最多 800 棵树、学习率 0.03。没有自动参数档位。
LSTM 直接输出 P 个小时；XGBoost 保存 P 个独立回归器组成的模型包。
XGBoost 使用 `xgboost_n_estimators` 和学习率，不提供 epoch 或 batch 参数。
可通过 `training.defaults`、`training.dataset_overrides`、`training.zone_overrides`
逐层覆盖；CLI 训练参数优先级最高。配置完整示例见 [README](../README.md) 和
`config.example.yaml`。`training.seed` 决定训练随机性，`run.experiment_seeds` 不会触发重新训练。

`forecaster` 和 `full` 阶段会先为选中区域补齐所需后端的离线模型，再运行预测矩阵。
已缓存模型直接复用；显式 `--force-train` 创建新版本，保留已被预测产物引用的旧版本。
`agent` 阶段只读取已有预测产物及其固定权重引用，不执行训练。

训练及早停验证覆盖的历史范围可能包含历史预测期间。输出记录每个区域的
`model_id`、训练起止时间及逐小时 `training_period_overlap`。这类结果不能解释为
按预测日截断训练的严格样本外回测；零重叠也不能排除使用了
预测日之后的训练信息。评估按实际模型版本分组，避免混合不同版本。

## 并行运行

将同一个 `run.data_dir` 改为列表，仍然运行 `python main.py`：

```yaml
data:
  adapter: auto
run:
  data_dir:
    - data/UrbanEV
    - data/CHARGED/JHB
    - data/MP-EVData
  max_parallel_datasets: 2
  output_folder: output
  forecast_starts: null
  zones: null
  forecast_models: [lstm]
  pipeline_stage: forecaster
```

每个文件夹用独立 Python 子进程，最多同时运行 `max_parallel_datasets` 个任务。
各进程使用对应数据集及区域的训练覆盖配置；一个数据集内部依次执行模型/种子/模式矩阵。
`pipeline_stage: train` 只准备离线权重；`forecaster` 生成预测及校准产物；`full` 执行完整闭环；
`dry_run: true` 使用确定性离线 Agent，不调用 API。
单任务出错时其他任务继续执行，最终进程返回失败并在清单中记录各自退出码。

多模型共用机器资源，`max_parallel_datasets` 可调整并发数量。
`training.lstm_workers` 和 `training.xgboost_workers` 分别控制每个数据集内各后端的
区域模型训练并发数，默认 1 和 2，可设为任意正整数。不存在跨数据集的全局训练锁；
同时运行的数据集数与各自的 worker 数相乘，决定该后端可能同时训练的模型数。
各 XGBoost 默认使用 2 个主机线程，可用 `training.defaults.xgboost_n_jobs` 覆盖。
CLI 仍使用 `--training-lstm-workers` 和 `--training-xgboost-workers`。

## 缓存与结果隔离

每个数据目录的 `cache/` 保存指纹缓存、特征、时间划分和离线权重。

```text
<data_dir>/cache/weight/<backend>/index.json
<data_dir>/cache/weight/<backend>/<safe-zone-key>/current.json
<data_dir>/cache/weight/<backend>/<safe-zone-key>/<model_id>/
    manifest.json
    model.pt 或 model.json
    preprocess.json
    training_history.csv
```

模型指纹来自当前区域的实际训练内容、特征和参数，其他区域数据改变不使本区域模型失效。
加载时校验清单及每个文件的哈希，损坏或缺失时明确报错，不在预测期间静默训练。
`train` 或实验开始前的准备阶段会为缺失、损坏的缓存创建新版本并记录原因；已有预测的
Agent-only 重放仍严格绑定原版本，原版本损坏时直接失败。
随机选区的资格缓存由 `zone_eligibility.load_zone_eligibility` 读取或生成，默认路径为：

```text
<data_dir>/cache/zone_selection_manifest.json
<data_dir>/cache/datasets/<dataset-fingerprint>/zone_selection/<policy-hash>/zone_eligibility.json
<data_dir>/cache/datasets/<dataset-fingerprint>/zone_selection/<policy-hash>/zone_eligibility.csv
```

`zone_selection_manifest.json` 是资格缓存索引。JSON 包含 `eligible_zone_ids`、
`excluded_zone_ids` 和排除原因；CSV 方便逐区域查看资格和统计结果，也记录适配器上游
已经排除的区域及其原因，例如非正电价、缺失电价或换电次数不属于电量。
后续使用同一组条件时直接读取已有资格缓存，再按照随机种子抽样。
数据指纹、预测日期、历史和验证长度、判定规则等变化会使用独立的缓存键，避免误用旧名单。
更改随机种子不改变区域资格，不需要重新计算这份名单。

单个数据集和多数据集运行使用相同的结果目录结构：

```text
<output_folder>/<experiment_name>/
├── UrbanEV/
├── CHARGED_JHB/
├── MP-EVData/
├── batch_manifest.json       # 多数据集调度记录
└── logs/                     # 各数据集的调度日志
```

`run.experiment_name` 决定实验目录名称，未填写时固定使用 `experiment`。
不再创建以程序启动时间命名的外层目录。各数据集目录内保留预测起点、种子、
模式、混合系数和模型等实验子目录；其中的日期代表预测起点，不是程序启动时间。
`experiment_runs.csv` 等汇总文件也分别写入各数据集目录。
单次运行始终包含 `blend_<alpha>/<backend>/`，混合系数使用六位小数精度，
相近值不会互相覆盖。全局 `diurnal_blend_alphas` 默认 `[0.0]`，
非 `null` 的模型独立配置（例如 `lstm_diurnal_blend_alphas`）完全覆盖全局配置；
每个模型只展开自己的有效值列表。CLI 同名复数选项先覆盖 YAML，再应用独立优先规则。

实验根目录中的隐藏文件 `.dataset_directories.json` 将目录名称绑定到实际数据路径。
默认目录名称如上；只有不同位置的数据源发生同名冲突时，后来的数据源才会追加
稳定的路径散列，避免混用模型和结果。多数据集运行的 `batch_manifest.json`
记录数据路径、启动命令、日志和退出码，日志直接放在实验目录的 `logs/` 中。
批次调度和各数据集输出使用独立进程锁，防止并发写入冲突。

使用相同实验名重新运行会复用目录，并更新同名产物、调度清单和日志。
若需要保留两次运行的独立结果，请修改 `experiment_name`。
旧结果保留在原位置，不会自动迁移；数据目录下的缓存位置保持不变。

列表运行时，`data.cache_dir`、`run.forecaster_output_dir`、
`run.agent_output_dir`、`run.precomputed_window_data` 保持 `null`，
避免多个数据集共用显式覆盖路径；程序会检查这一点。

重用模型时，将 `pipeline_stage` 从 `forecaster` 改为 `agent`，保持
`output_folder`、`experiment_name` 和其他实验设置一致即可；单数据集和多数据集
均按相同规则查找，无需再将 `output_folder` 指向某个时间批次目录。
预测起点下的 `forecaster` 目录保存预测、Bias 和离线权重引用；多模式 `full` 阶段的这些产物
位于各自模式目录，权重仍共用数据集缓存。Agent 阶段优先读取对应模式目录中的 `forecaster_manifest.json`，
没有时再读取共享模型目录，支持这两种来源。
使用单个 `data_dir` 时，可以通过 `forecaster_output_dir` 精确指定当前协议模型的
`forecaster` 目录或 `forecaster_manifest.json` 文件。旧协议的预测器产物需要重新生成，
不提供旧模型迁移或回放；历史结果字段兼容仅在离线评估读取边界处理。
Agent 阶段校验适配器、数据目录和数据指纹，拒绝混用其他数据集的模型。

## 数据语义和质量策略

当前闭环要求严格正的基准电价，因此不会把未知或零电价替换为任意常数。
每个缓存 `feature_manifest.json` 中有 `excluded_sites`、单位、日期范围和处理规则。
原始文件完整保留，筛选发生在适配层，预测和闭环使用相同的可用站点集合。

**CHARGED**：以每个站点为 `zone_id`，保留小时 kWh、电价和服务费；支持原始表中的
`Unnamed: 0` 时间列，以及 `site/site_id` 两种静态标识。整个发布期间电价缺失、非正，
或电量不合法的站点不进入当前流程。并不是所有公开站点都可用于定价实验。
各城市价格保留来源单位；AMS 上游 README 对币种的文字标注存在疑点，本项目不推断
币种或汇率，跨城市应比较相对调价幅度。静态表中的全期间电量、时长和平均功率
不进入模型或 Agent，避免把未来目标统计当作静态特征。POI 对所有原始站点分配后
再筛选，避免零电价站点被排除后将整座城市的 POI 归给少数剩余站点。

**MP-EVData**：读取官方 `station-level load Profile 1h.xlsx` 和 `price.xlsx`。
将每小时平均 kW 乘以 1 小时得到 kWh；按月份和时段展开电价，区间左闭右开，
跨午夜正确拆分，Sharp 优先于与其重叠的 Peak。仅使用 2024 年记录，丢弃文件末尾
2025 年的记录，避免套用没有发布的 2025 电价。实际可用站点为 A1/A2/A3/A6/A10。

- A4、A9：电价表为空，站点元数据标为 Free。
- A5：1 月、12 月缺少 21:00–22:00 电价，保守地排除整个站点，不填补未知价格。
- A7、A8：`session_count` 是换电次数，不是电量。
- A10：实测记录从 2024-09-27 开始。自动选区时多个起点按全部可用区域共同覆盖取点，
  日期受其覆盖范围约束，并预留训练与验证历史；显式选区包含 A10 时也受此约束。
  如果手动指定更早起点，需通过 `run.zones`
  排除尚未投运的站点。
- 不使用有歧义的合并容量单元格或全年的订单总数；保留真实站点类型。
- 没有天气数据时天气摘要为缺失，不伪造温湿度或降雨。

这里的零价/覆盖筛选是数据集可用性筛选，不表示在完整原始城市样本上的结果。
闭环成功仍是模型预测条件下的结果，不等同于真实电价干预的因果效果。

## 验证

```powershell
python -B -m unittest discover -s tests -p test_datasets.py -v
python -B -m unittest discover -s tests -p test_evaluate.py -v
```

新增测试覆盖价表尖峰优先级、跨午夜、冲突拒绝、换电/免费站点剔除、负荷单位、
晚投运站点、源文件不变、前导零站点 ID 缓存往返、配置切换、输出隔离及跨进程锁。
