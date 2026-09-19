# 多数据集训练

项目支持 UrbanEV、CHARGED 和 MP-EVData。原有不带 `--dataset` 的命令和
UrbanEV 数据保持兼容；使用 `--dataset` 可启用独立路径与合适的预测日期。

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

## 切换训练

```powershell
python main.py --dataset urbanev --forecast-model AR --diurnal-blend-alpha 0 --pipeline-stage forecaster
python main.py --dataset charged --city JHB --forecast-model AR --diurnal-blend-alpha 0 --pipeline-stage forecaster
python main.py --dataset mp_evdata --forecast-model AR --diurnal-blend-alpha 0 --pipeline-stage forecaster
```

`--dataset mp-evdata` 也是合法写法。CHARGED 支持 AMS/JHB/LOA/MEL/SPO/SZH，默认 JHB。
将 `AR` 换成 `lstm`、`chronos` 或 `timesfm` 使用其他现有后端。
`--device cpu/cuda/auto` 覆盖 LSTM/Chronos 设备，`--lstm-epochs N` 覆盖训练轮数。
TimesFM 沿用它自身的设备选择逻辑。

| 数据集 | 原始文件目录 | 默认预测起点 | 默认输出目录 |
|---|---|---|---|
| UrbanEV | `data/UrbanEV` | 2022-10-14 | `output/urbanev` |
| CHARGED | `data/CHARGED/<city>` | 2023-06-01 | `output/charged/<city>` |
| MP-EVData | `data/MP-EVData` | 2024-11-01 | `output/mp_evdata` |

每个数据目录的 `cache/` 保存自己的指纹缓存、特征和时间划分；模型及评估输出
保存在对应输出目录内。`--output-folder` 指定的是公共根目录，仍会追加数据集/
城市子目录。直接使用新适配器 `--dataset-adapter charged/mp_evdata` 也会隔离输出，
但不会自动切换路径、时间或站点；通常应使用 `--dataset`。

配置优先级：基础 `run` 的模型训练参数 → 数据集默认路径/日期 →
`datasets.<name>.run/data` → CLI 参数。切换会清除旧数据集的区域列表、日期列表、
数据映射、缓存路径及旧模型/窗口文件引用。可在 `config.yaml` 中显式覆盖，例如：

```yaml
datasets:
  mp_evdata:
    run:
      forecast_start: "2024-11-01 00:00:00"
      zones: ["A1", "A2", "A3", "A6", "A10"]
      horizon_days: 2
      lstm_epochs: 50
```

没有 `--dataset` 的原有 UrbanEV 命令沿用原先输出路径；新命令不会复用或覆盖旧输出。
直接配置 `--forecaster-output-dir` 或 `--agent-output-dir` 是显式路径覆盖，应为不同
数据集设置不同目录。Agent 阶段检查适配器和数据指纹，拒绝混用其他数据集的模型。

## 并行训练

```powershell
python train_datasets.py --datasets urbanev charged:JHB mp_evdata --models lstm --max-workers 3 --device cpu
# 多城市；每个数据集进程内部依次运行所列模型
python train_datasets.py --datasets charged:JHB charged:LOA mp_evdata --models AR lstm --max-workers 2
```

每个数据集/城市用独立 Python 子进程。默认只训练 forecaster，不调用 Agent API。
默认模型是 AR；`--models` 可指定多个后端。每批创建唯一的
`output/parallel/<UTC时间>_<随机ID>/`，包含各数据集日志和 `batch_manifest.json`，
后者记录命令、退出码和整批状态。单个任务失败会记录非零退出码，其他任务继续完成。
重复的数据集/城市任务会拒绝启动，输出目录也有进程锁，防止同时覆盖。

`--max-workers` 控制同时运行的数据集数量。多个 GPU 模型会共享显存；单卡显存不足时
降低并发数或用 `--device cpu`。这不是多 GPU 自动分配器。

完整闭环离线验证：

```powershell
python train_datasets.py --datasets urbanev charged:LOA mp_evdata --models AR --max-workers 3 --pipeline-stage full --dry-run
```

正式 Agent 实验去掉 `--dry-run`，使用已有 `config.yaml` 的 Agent 配置。
单独重用某数据集 forecaster 时保持相同的模型、起点、区域和输出根目录，例如：

```powershell
python main.py --dataset mp_evdata --forecast-model AR --diurnal-blend-alpha 0 --pipeline-stage agent --dry-run
```

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
- A10：实测记录从 2024-09-27 开始。默认起点设为 11 月以保证历史窗口可用；
  如果使用更早起点，需通过 `--zones` 排除尚未投运的站点。
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
