# 地方债回测框架收益口径说明

## 目标

当前框架用于检验地方债看板对 10Y 地方债久期敞口的择时能力。它不是具体个券回测，也不是用真实票面利率、应计利息、净价、全价逐笔复原的持有期收益。

## 标的

第一版标的为 Choice 指标 `E1704597`：

- 名称：地方政府债到期收益率:10年
- 单位：%
- 来源：中证指数公司

回测将这条收益率曲线转成 10Y 地方债的合成总收益。

## 收益合成方法

日收益拆成两部分：

```text
total_return = carry_return + duration_pnl
carry_return = previous_yield / 252
duration_pnl = -modified_duration * yield_change
```

其中：

- `previous_yield`：上一可得交易日的 10Y 地方债到期收益率，小数口径。
- `yield_change`：当日收益率减上一可得交易日收益率，小数口径。
- `modified_duration`：用 10 年平价债、年付息、上一期收益率估算的修正久期。

策略回测会进一步把这两部分按仓位拆到日度归因：

```text
strategy_carry_return = position * carry_return
strategy_capital_return = position * duration_pnl
carry_excess_return = strategy_carry_return - carry_return
capital_excess_return = strategy_capital_return - duration_pnl
```

这组字段用于判断低仓位或空仓是否值得：低仓位通常会少吃 `carry_return`，如果少吃的票息不能被 `capital_excess_return` 弥补，那么策略即使回撤较小，也可能长期跑不赢满仓基准。

上述 `duration_pnl` 是传统净值中的久期折算价格收益率，不作为资本利得 BP。交易评价使用不乘久期的收益率方向变动：

```text
strategy_capital_bp = -position * asset_yield_change_bp
benchmark_capital_bp = -benchmark_yield_change_bp
capital_excess_bp = strategy_capital_bp - benchmark_capital_bp
```

因此，多头遇到收益率下行、空头遇到收益率上行时资本利得 BP 为正。这里的 1BP 就是收益率曲线变动 1BP，不再表示价格收益率的万分之一。

## 绩效指标口径

当前绩效指标按以下口径计算：

```text
累计收益率 = 期末净值 / 期初净值 - 1
年化收益率 = (期末净值 / 期初净值) ^ (252 / 有效交易日间隔) - 1
年化波动率 = 日收益率标准差 * sqrt(252)
超额年化收益率 = 年化收益率 - 无风险利率
夏普比率 = 超额年化收益率 / 年化波动率
最大回撤 = 净值 / 历史最高净值 - 1 的最小值
Calmar比率 = 年化收益率 / abs(最大回撤)
```

无风险利率第一版使用定值 `1.4%` 年化，近似资金市场无风险收益水平。后续如果接入 DR001 日度序列，可以替换为浮动日度无风险利率，再按日度超额收益计算夏普。

报告中保留 `日胜率_参考`，计算方式是日收益率大于 0 的交易日占比。它只反映日度涨跌分布，不代表策略每次调仓是否成功。

对周度看板策略，更有解释力的是 `调仓周期胜率`：

```text
单个调仓周期收益 = 该 signal_date 持有到下一次 signal_date 期间的日收益复利
调仓周期胜率 = 单个调仓周期收益 > 0 的周期数 / 全部调仓周期数
单周期平均收益 = 所有调仓周期收益的简单平均
```

调仓周期胜率用于判断每周信号方向，但不是产品实际交易胜率。交易盘评价应优先看开仓到平仓的已平仓交易胜率、单笔资本利得BP、盈亏比、最差交易和资本利得回撤；周度周期指标作为错判归因的辅助信息。

## 权重搜索目标函数

`dashboard_weight_search_v1` 只调整因子权重，不调整定性阈值。权重以 5 分为最小步长；正向权重合计固定为 100；发飞惩罚项在 0 到 -30 之间搜索。

## 参数配置与本地网页工具

策略参数现在可以放在 `configs/*.json` 中管理。JSON 文件记录四类参数：

```text
weights：因子权重
thresholds：定性阈值
positions：仓位规则
objective：搜索目标函数权重
backtest：单次回测起止日期
```

正式配置使用 `configs/` 下的中文策略名保存；旧配置与研究中间配置分别放在 `configs/archive/` 和 `configs/experiments/`。网页选择的基线配置就是本次手动调参起点。

本地网页调参工具为：

```text
app_backtest_dashboard.py
```

启动方式：

```powershell
cd "C:\Documents\CodeArchive\工作\2026中信证券code\地方债"
streamlit run app_backtest_dashboard.py
```

如果系统命令找不到 `streamlit`，可以使用：

```powershell
C:\Users\chris\AppData\Local\Programs\Python\Python313\python.exe -m streamlit run app_backtest_dashboard.py
```

网页左侧可以调整权重、阈值、仓位和单次回测区间；点击“运行回测”后，在右侧查看资本利得交易、净值、归因、绩效、错判周期和最新信号。点击“保存配置”会把当前页面参数保存到 `configs/`，后续可以重新加载。目标函数在顶部“搜索研究”页集中调整，并同时传给权重搜索和阈值搜索；单次回测页只读展示配置中保存的目标函数。

“搜索研究”先选择一个基线 JSON，再允许临时调整因子权重、定性阈值、仓位与执行规则。两类搜索的参数口径不同：

- 因子权重搜索：遍历九因子权重，固定页面当前的定性阈值和仓位规则；页面中的基线权重只作为对照，不限制候选空间。
- 定性阈值搜索：遍历定性阈值、看空总分和看空确认规则，固定页面当前的因子权重及多/中/空仓位。
- 两类搜索均强制使用当前数据源的完整历史，不继承单次回测配置中的日期窗口；搜索得到的最优配置和实验归档会记录页面实际参数。

策略仓位规则集中维护在 `strategies/position_policy.py`：

```text
总分 >= bullish_threshold -> bullish_position
总分 < bearish_threshold -> bearish_position
其他 -> neutral_position
```

当前默认值为：

```text
bullish_threshold = 70
bearish_threshold = 30
bullish_position = 1.0
neutral_position = 0.7
bearish_position = -1.0
```

默认排序目标函数为：

```text
objective = 累计资本利得BP + 0.25 * 资本利得超额BP + 10 * 已平仓交易胜率
资本利得超额BP = 策略累计资本利得BP - 满仓基准累计资本利得BP
```

`all_results.csv` 和 `top_configs.csv` 中的 `objective` 是候选排序分数，不是收益率或BP本身。网页可以修改各项权重；搜索报告会记录本次实际使用的目标函数。

## 资本利得交易定义

逐笔交易与周度错判周期是两个不同口径：

```text
仓位 0 -> 非0：开仓
仓位 非0 -> 0：平仓
多头 -> 空头或空头 -> 多头：平掉原交易并反向开新交易
同方向仓位增减：仍视为同一笔交易
```

样本结束时仍未平仓的交易计入累计资本利得和未平仓浮动BP，但不计入已平仓胜率。满仓基准从样本开始一直持有，因此固定为开仓1笔、已平仓0笔、当前未平仓1笔，胜率留空。周度信号周期只用于错判归因，不再当作交易笔数。

## 网页运行与实验归档

网页点击“运行回测”后，除了在当前会话展示结果，还会自动在以下目录生成不可覆盖的运行记录：

```text
backtest_outputs/experiments/
  YYYYMMDD_HHMMSS__策略名称/
```

每次运行同时保存：

- `config.json`：本次实际使用的完整参数快照；
- `run_manifest.json`：运行时间、数据文件哈希、代码文件哈希和主要绩效；
- `performance_report.html`：完整HTML报告；
- `performance_metrics.csv`：策略与基准绩效；
- `period_diagnostics.csv`：逐调仓周期诊断；
- `signal_score.csv`：因子得分与仓位；
- `strategy_nav.csv`：净值与收益归因。

因此，策略JSON负责定义“怎么运行”，实验目录负责记录“当时用哪版数据和代码跑出了什么结果”。即使以后只通过网页运行，也不能只保留当前页面状态。

网页“历史实验”表中每次运行都有“查看”入口，打开后使用当前 Streamlit 组件恢复归档结果，不重新运行当前数据。权重搜索和阈值搜索完成后都会归档最优策略；权重搜索还会按九项最优权重命名并保存配置 JSON。

首页“研究快照”只从当前数据源完整历史区间一致的归档中选择代表策略。完整区间由周度信号首日与基准曲线末日动态计算；数据扩展后，旧的较短归档不会继续参与首页最佳策略比较。

历史实验表的每一行都有“查看”入口。进入历史结果后，左侧恢复完整策略控制台，可从归档权重、阈值、仓位和回测区间继续修改；再次运行会新增实验，不覆盖原归档。回测区间首日作为净值锚点，策略收益、基准收益和资本利得 BP 均设为 0，不带入起始日前一交易日的市场变动。

## 看空确认规则

仓位配置除总分阈值外，还支持三项可复现的看空约束：

```text
bearish_min_core_factors
bearish_require_supply_or_demand
bearish_confirmation_periods
```

- `bearish_min_core_factors`：执行看空前至少需要多少个核心模块利空；0表示不限制。
- `bearish_require_supply_or_demand`：1表示看空信号必须包含供给或银行需求模块利空。
- `bearish_confirmation_periods`：低分及确认条件连续满足多少周后才执行看空。

核心模块分为供给、银行需求、估值和非银情绪四组。默认值为 `0 / 0 / 1`，与增加确认规则前的V1行为一致。

## 阈值调参实验V1

运行入口：

```powershell
python run_backtest.py --task dashboard_threshold_search_v1
```

该实验固定最初版九因子权重，并固定看多、中性、看空仓位为 `1 / 1 / -1`。候选阈值、总分门槛和确认规则使用NumPy矩阵批量计算，避免逐候选循环运行完整回测。

结果保存在：

```text
backtest_outputs/阈值调参实验_v1/
```

当前短样本结果显示，单独修改定性分位阈值没有优于原始阈值的证据。样本内最佳改善来自看空触发规则：总分低于35、至少两个核心模块利空，并且必须包含供给或需求利空。该结论仍需等待历史数据扩展后进行滚动样本外验证。

所以搜索结果中的“最好”不是单纯按总收益率最高选出，而是默认同时考虑：

- 策略累计资本利得BP；
- 相对满仓基准的资本利得超额BP；
- 已平仓交易胜率。

这里的资本利得 BP 均为 `-仓位 × YTM变化_BP`，不乘久期。旧实验若使用久期折算价格收益 BP，会在历史表中标记为“旧口径”，不能与新搜索结果直接横向排序。

搜索研究页仍可加入资本利得回撤BP、传统收益、夏普和调仓周期胜率等辅助项。每次搜索报告应以页面实际参数为准。

## 调仓周期诊断

策略报告会额外输出 `period_diagnostics.csv`，并在 HTML 中展示错判类型统计和拖累最大的调仓周期。每个调仓周期按 `signal_date` 分组，计算：

```text
周期策略收益 = 周期内 strategy_return 复利
周期基准收益 = 周期内 total_return 复利
周期超额收益 = 周期策略收益 - 周期基准收益
周期carry贡献 = 周期内 strategy_carry_return 简单求和
周期资本利得贡献 = 周期内 strategy_capital_return 简单求和
```

错判类型第一版定义：

- 做空做反：仓位 < 0，基准收益 > 0，策略收益 < 0。
- 低仓少吃：0 <= 仓位 < 1，基准收益 > 0，策略跑输基准。
- 该空没空：仓位 >= 0，基准收益 < 0，策略未显著规避损失。
- 有效防守：仓位 < 1，基准收益 < 0，策略跑赢基准。
- 有效进攻：仓位 > 0，基准收益 > 0，策略赚钱。

## 为什么可以用 YTM 近似 carry

当前没有具体债券的票面利率、净价、全价和应计利息，因此不能精确计算真实票息收入。用 `YTM / 252` 作为 carry，是在构造 constant-maturity 10Y 久期敞口时的近似做法。

这个近似隐含：

- 每天持有的是一只接近 10 年剩余期限的代表性平价债。
- 代表性债券的票息大致接近当时的市场到期收益率。
- 回测关注的是 10Y 地方债利率方向和久期敞口择时，而不是某一只个券的真实现金流。

所以它适合回答：

```text
看板信号是否能帮助择时 10Y 地方债久期敞口？
```

不适合回答：

```text
买入某一只地方债并持有，真实票息、净价、全价和应计利息收益是多少？
```

## 当前口径的局限

- 没有真实票面利率。
- 没有应计利息。
- 没有净价/全价。
- 没有真实指数久期。
- 没有滚降收益。
- 没有交易成本、融资成本和做空工具约束。
- 看空仓位是方向性风险敞口假设，不代表已经落实到具体可交易工具。

## 后续升级路线

1. 曲线近似 v2：加入 roll-down。
2. 指数净值法：用地方债指数净值直接作为标的收益。
3. 个券组合法：用一篮子 10Y 附近地方债的全价、应计利息、票面利率和久期做真实总回报。
