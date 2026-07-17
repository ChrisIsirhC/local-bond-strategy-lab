# 10Y地方债策略实验室

Streamlit 研究工具，用周度看板信号回测10Y地方债久期交易，主要评价资本利得BP、逐笔胜率和资本利得回撤。

## 启动

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app_backtest_dashboard.py
```

## 部署入口

- 主文件：`app_backtest_dashboard.py`
- Python：3.12或3.13
- 启动命令：`streamlit run app_backtest_dashboard.py`

## 数据边界

部署包只包含应用运行必需的处理后周度看板、10Y地方债收益率曲线、策略配置和少量代表性回测结果。不包含原始数据、Choice/RQData取数脚本、账号、密码或API密钥。

## 公开历史案例

网站历史页只保留以下三个经典案例，配置文件也使用对应的中文名称：

- 基础配置·100-激进看多·日频·训练截止20250101·联合搜索
- 基础配置·100-激进看多·日频·止损5BP·训练截止20250101·联合搜索
- 基础配置·100-重胜率-权重搜索最优·周频·训练截止20250101·联合搜索

完整实验仍保留在本地研究项目中，不随公开部署包发布。

免费Streamlit实例的本地文件系统不是永久存储。网页中新生成的配置和实验在实例重启后可能消失；需要长期保存时应接入数据库或对象存储。
