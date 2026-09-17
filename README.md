# 10Y地方债策略实验室

Streamlit 研究工具，用周度看板信号回测 10Y 地方债久期交易，主要评价资本利得 BP、逐笔胜率和资本利得回撤。

## 启动

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app_backtest_dashboard.py
```

## 部署入口

- 主文件：`app_backtest_dashboard.py`
- Python：3.12 或 3.13
- 启动命令：`streamlit run app_backtest_dashboard.py`

## 数据边界

部署包包含应用运行必需的处理后周度看板、10Y 地方债收益率曲线、策略配置及完整历史实验归档。不包含原始数据、Choice/RQData 取数脚本、账号、密码或 API 密钥。

## 公开历史实验

历史页公开部署包中包含的全部历史实验；每条策略保留不可变运行 ID、配置快照、信号、交易明细、绩效指标和研究溯源。

免费 Streamlit 实例的本地文件系统不是永久存储。网页中新生成的配置和实验会在实例重启后丢失；需要长期保存时应接入数据库或对象存储。
