# IndexTrack MVP 验收清单（DoD）

对应文档：`requirement.md` 第 9 节

## 1. 一条命令可运行并输出分析报告
- 状态：✅ 已完成
- 运行命令：
  - `PYTHONPATH=src python3 -m indextrack.cli --index SP500 --period 1M`
- 结果：可输出数据状态、三档概率、中文结论与风险提示。

## 2. 可分析 S&P 500 与纳斯达克
- 状态：✅ 已完成
- 实现说明：
  - 指数映射支持 `SP500` 与 `NASDAQ`。
  - CLI 支持 `--index SP500|NASDAQ|BOTH`。

## 3. 概率覆盖短/中/长三个时间尺度
- 状态：✅ 已完成
- 实现说明：
  - `ScenarioEngine` 输出 `short/mid/long` 三组 `上涨/震荡/下跌` 概率。
  - `tests/test_scenario.py` 验证概率合法性（范围 + 总和）。

## 4. 每次运行默认尝试拉取最新数据，并显示更新时间
- 状态：✅ 已完成
- 实现说明：
  - 每次 CLI 调用先走主备数据源拉取，再进入分析。
  - 输出中包含 `last_trade_date` 与 `source`，并带新鲜度判断。

## 5. 数据不可更新时清晰告警并标注回退时间
- 状态：✅ 已完成
- 实现说明：
  - 主备失败后自动回退本地缓存。
  - 输出 `数据告警`，包含最近快照日期。
  - `tests/test_provider_fallback.py` 覆盖缓存回退路径。

## 6. 中文易读结论
- 状态：✅ 已完成
- 实现说明：
  - `ReportGenerator` 生成中文结论（趋势 + 概率 + 数据状态 + 免责声明）。
  - `tests/test_report_zh.py` 验证可读性和关键字段完整性。

## 7. 质量验证记录
- 单元测试：
  - `PYTHONPATH=src python3 -m unittest tests.test_provider_fallback -v`
  - `PYTHONPATH=src python3 -m unittest tests.test_scenario -v`
  - `PYTHONPATH=src python3 -m unittest tests.test_report_zh -v`
- 集成测试：
  - `PYTHONPATH=src python3 -m unittest tests.test_orchestrator -v`
- 语法检查：
  - `PYTHONPYCACHEPREFIX=/tmp/python-cache python3 -m compileall src tests`

## 8. 已知限制（不阻塞 MVP）
- 节假日日历覆盖 NYSE 常规休市日，但未覆盖临时停市（例如国家哀悼日等非常规事件）。
