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
  forecast_start: null
  forecast_starts: null
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

`forecast_start`、`forecast_starts` 都留空且 `forecast_start_count: 1`（默认）时，
使用当前数据最后 `horizon_days` 天作为预测窗口，此前数据作为训练历史。
`zones: null` 从当前数据自动选站。
显式填写日期或站点时会按你的配置执行，因此跨数据集切换时需使用该数据集的日期和编号。
模型、轮数、Agent 模式和其他训练参数仍使用原有 `run` 字段。
无需设置设备；后端自动选择可用设备。

每个数据集随机选择 5 个区域，可设置：

```yaml
run:
  zones: null
  experiment_zone_count: 5
  experiment_zone_selection: random
  experiment_zone_seed: 42
```

每个数据集先抽样一次，所有模型、Agent 模式和训练种子共用这组区域。
更改 `experiment_zone_seed` 可重新抽样；区域不重复，最多选择当前数据集的可用区域数。
显式 `zones` 或 `--zones` 优先于自动抽样。默认 `representative` 保持按代表性评分选区。

每个数据集自动选择两个均匀分布的预测起点，可增加：

```yaml
run:
  forecast_start: null
  forecast_starts: null
  forecast_start_count: 2
```

程序按天筛选有效起点：起点之前必须包含 `history_days + validation_days` 天的
连续小时记录，之后必须包含完整的 `horizon_days` 天预测序列，再取候选日期中约 1/3 和 2/3
位置的两个起点（均为 00:00）。这里的天数均按每天 24 小时计算。
随机选区时先固定区域，再按这些区域共同覆盖的时间筛选，两次实验共用同一组区域。
显式选区也按选中区域筛选；代表性选区则先按全部可用区域的共同覆盖筛选日期，
再使用第一个起点之前的数据选区。存在缺失小时的窗口不会作为候选，候选不足会明确报错。
`forecast_start_count` 可设为更大的正整数；手动指定 `forecast_start`、`forecast_starts`
或对应 CLI 日期时，手动日期优先，不再自动取点。

原先新增的 `--dataset`、`--city`、`--device` 和 `datasets` 配置块已移除，
不再需要 `train_datasets.py`。原有模型、阶段等 CLI 参数仍可覆盖配置；
`--data-dir` 可临时指定单个文件夹。

## 并行训练

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
  forecast_start: null
  forecast_starts: null
  zones: null
  forecast_models: [lstm]
  pipeline_stage: forecaster
```

每个文件夹用独立 Python 子进程，最多同时运行 `max_parallel_datasets` 个任务。
各进程共享配置中的训练参数；一个数据集内部依次执行模型/种子/模式矩阵。
`pipeline_stage: forecaster` 只训练预测模型；`full` 执行完整闭环；
`dry_run: true` 使用确定性离线 Agent，不调用 API。
单任务出错时其他任务继续执行，最终进程返回失败并在清单中记录各自退出码。

多模型共用机器资源，`max_parallel_datasets` 可调整并发数量。
设备选择仍由各模型自动完成。

## 缓存与结果隔离

每个数据目录的 `cache/` 保存指纹缓存、特征和时间划分。
单个数据集的结果写入：

```text
<output_folder>/<adapter>/<folder>_<path-hash>/<原有实验和模型目录>
```

路径散列保证不同位置的同名文件夹也不会混用结果。
并行训练在上述结构外再增加 `<UTC时间>_<随机ID>/` 批次目录，
其中的 `batch_manifest.json` 记录数据路径、启动命令、日志和退出码。
旧结果保留在原位置，新运行采用上述目录结构。
每个输出目录都有进程锁，防止两个进程同时写入。

列表运行时，`data.cache_dir`、`run.forecaster_output_dir`、
`run.agent_output_dir`、`run.precomputed_window_data` 保持 `null`，
避免多个数据集共用显式覆盖路径；程序会检查这一点。

单个数据集重用模型时，将 `pipeline_stage` 改为 `agent`，其他实验设置保持一致。
并行批次重用模型时，还需将 `output_folder` 指向之前的批次目录；
子进程会从该目录下各自的数据集输出中读取，新的调度日志放在独立子目录。
也可以使用单个 `data_dir`，并通过 `forecaster_output_dir` 精确指定旧模型目录。
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
- A10：实测记录从 2024-09-27 开始。自动选择多个起点且选中 A10 时，日期受其
  覆盖范围约束，并预留训练与验证历史；如果手动指定更早起点，需通过 `run.zones`
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
