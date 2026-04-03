# IndexTrack

IndexTrack 是一个本地工具，可分析 `S&P 500` 与 `Nasdaq Composite`，输出短/中/长期趋势概率和中文结论。

## 先跑起来（中文）

如果你只想先看到结果，按下面 4 步做就行。

### 1. 进入项目目录
```bash
cd /Users/nickge/Documents/CODE/Project/IndexTrack
```

### 2. 准备环境（只需首次）
需要：
- Python 3.9+
- `uv`

如果你没有 `uv`：
```bash
python3 -m pip install uv
```

安装依赖：
```bash
UV_CACHE_DIR=/tmp/uv-cache python3 -m uv sync
```

### 3. 命令行模式（CLI）运行分析
```bash
PYTHONPATH=src python3 -m indextrack.cli --index BOTH --period 1M
```

你会看到：
- 数据状态（数据源、最新交易日、新鲜度）
- 短期/中期/长期三档概率（上涨 / 下跌 / 不确定）
- 中文结论与风险提示

推荐新模型（分位数 + OOF 校准）：
```bash
PYTHONPATH=src python3 -m indextrack.cli --index BOTH --period 1Y --model quantile
```

当前默认采用 `horizon-specific pipeline`：
- `short`: `raw + calibration + shrinkage`（保留 full chain）
- `mid`: `raw only`（默认关闭 calibration）
- `long`: `raw only`（默认关闭 calibration）

输出 quantile 诊断信息（train/oof 标签分布、raw/cal/final、可靠性）：
```bash
PYTHONPATH=src python3 -m indextrack.cli --index SP500 --period 1Y --model quantile --prob-debug
```

运行 horizon-specific 消融实验（ablation，含 short/mid/long pipeline、long threshold/sigma_floor/lambda、recent calibration、prior adjustment）：
```bash
PYTHONPATH=src python3 -m indextrack.cli --index SP500 --period 1Y --model quantile --prob-ablation
```
输出为 `short/mid/long` 分开实验与各 horizon 推荐方案。

导出完整诊断包（summary + detailed + sample csv + ablation + json）：
```bash
PYTHONPATH=src .venv/bin/python scripts/export_quantile_diagnostics.py --symbol SP500 --period 3Y
```
导出后可直接打开固定入口文件：
- `.data/diagnostics/diagnostics_latest.md`

### 4. 页面模式（UI）查看图表
启动 UI 服务：
```bash
PYTHONPATH=src python3 -m indextrack.cli --ui --ui-host 127.0.0.1 --ui-port 8000
```

浏览器打开：
- `http://127.0.0.1:8000/`

可切换图表周期：
- `http://127.0.0.1:8000/?period=1M`
- `http://127.0.0.1:8000/?period=3M`
- `http://127.0.0.1:8000/?period=6M`
- `http://127.0.0.1:8000/?period=1Y`

可切换分析模型（UI 顶部有模型页签）：
- `http://127.0.0.1:8000/?period=1Y&model=quantile`
- `http://127.0.0.1:8000/?period=1Y&model=legacy`

页面中每个指数卡片会展示：
- 趋势图
- 图下“短期/中期/长期”的上涨 / 下跌 / 不确定概率
- raw probability vs calibrated probability
- horizon 阈值、当前 regime、signal_strength 分数
- headline 文案分级（`上行/偏上/不确定/偏下/下行`），避免“最大概率即绝对方向”
- 折叠区中的数据来源与分析方法

停止 UI：
- 回到终端按 `Ctrl+C`

## 免费公网部署（Render，推荐）

项目已补齐部署文件：`Dockerfile`、`render.yaml`、`scripts/start_web.py`、`requirements.txt`。  
你现在可以直接部署到 Render（若你的账号当前可用 free web 实例，优先选 free）。

### 中文部署步骤

1. 把当前项目推送到你的 GitHub 仓库。  
2. 打开 Render，选择 `New +` -> `Blueprint`，连接这个仓库。  
3. Render 会自动读取根目录的 `render.yaml`。  
4. 部署完成后，打开：
   - `https://<你的服务域名>/`
5. 健康检查地址：
   - `https://<你的服务域名>/healthz`

可选环境变量（在 Render 控制台里配置）：
- `INDEXTRACK_DEPLOY_MODEL=quantile`（或 `legacy`）
- `INDEXTRACK_DEPLOY_PERIOD=1Y`
- `INDEXTRACK_TIMEZONE=America/New_York`
- `INDEXTRACK_DB_PATH=.data/indextrack.db`

本地先验收 Docker（可选）：
```bash
docker build -t indextrack-web .
docker run --rm -p 8000:7860 indextrack-web
```
打开：`http://127.0.0.1:8000/`

### English Deployment Steps

1. Push this project to your GitHub repository.  
2. In Render, click `New +` -> `Blueprint`, then connect the repo.  
3. Render will auto-detect `render.yaml` in the root directory.  
4. After deploy, open:
   - `https://<your-service-domain>/`
5. Health check endpoint:
   - `https://<your-service-domain>/healthz`

Optional environment variables:
- `INDEXTRACK_DEPLOY_MODEL=quantile` (or `legacy`)
- `INDEXTRACK_DEPLOY_PERIOD=1Y`
- `INDEXTRACK_TIMEZONE=America/New_York`
- `INDEXTRACK_DB_PATH=.data/indextrack.db`

## 常用命令

只看 S&P 500：
```bash
PYTHONPATH=src python3 -m indextrack.cli --index SP500 --period 3M
```

只看 Nasdaq：
```bash
PYTHONPATH=src python3 -m indextrack.cli --index NASDAQ --period 6M
```

查看帮助：
```bash
PYTHONPATH=src python3 -m indextrack.cli --help
```

## 参数说明

- `--index SP500|NASDAQ|BOTH`
- `--period 1M|3M|6M|1Y|3Y|5Y|90D`
- `--model quantile|legacy`（默认 `quantile`）
- `--prob-debug` 输出 quantile 调试信息（train/oof 标签分布、raw/cal/final 指标、reliability bins）
- `--prob-ablation` 输出 quantile horizon-specific 消融实验表格与推荐方案
- `--ui` 启动本地 UI
- `--ui-host` UI 监听地址，默认 `127.0.0.1`
- `--ui-port` UI 端口，默认 `8000`
- `--db-path` 本地缓存数据库，默认 `.data/indextrack.db`
- `--log-path` 日志文件，默认 `.data/indextrack.log`
- `--log-level INFO|WARNING|ERROR`
- `--timezone` 时区，默认 `America/New_York`

## 常见问题

报错 `不支持的 period`：
- 你输入了不支持的周期，改成 `1M/3M/6M/1Y/3Y/5Y/90D` 之一。

报错 `不支持的时区`：
- 用标准时区名，例如 `America/New_York`、`Asia/Shanghai`。

报错 `No module named 'numpy'`：
- 先同步项目依赖：`UV_CACHE_DIR=/tmp/uv-cache python3 -m uv sync`
- 若你使用的是已有虚拟环境，也可直接补装：`python3 -m pip install numpy`
- 未安装依赖时，CLI 会自动回退到 `--model legacy`，但建议尽快补齐依赖以启用 `quantile` 模型。

UI 结果看起来“没变化”或总是偏向同一方向：
- 先确认当前模型：展开卡片下方折叠区，查看“分析方法”中的 `当前模型: quantile/legacy`。
- 若显示 `legacy`，通常是因为缺少依赖或被显式设置了 `--model legacy` / `INDEXTRACK_MODEL=legacy`。
- 再确认数据时效：若出现“已回退到最近缓存数据”，说明本次未拉到新数据。
- 若你需要恢复旧版 quantile 参数行为，可设置 `INDEXTRACK_PROB_PROFILE=baseline` 做 A/B 对比。

为什么 mid/long 看起来没有 calibrated 变化：
- 这是当前默认策略（基于最新诊断）：`mid/long` 默认关闭 calibration，避免 regime shift 下的反向放大。
- 如需强制打开，可设置：
  - `INDEXTRACK_PROB_CALIB_MODE_20=conservative`
  - `INDEXTRACK_PROB_CALIB_MODE_60=conservative`
  - 并配合 `INDEXTRACK_PROB_CALIB_BLEND_20/_60` 调整强度。

为什么不会再直接输出 99%/100% 的方向概率：
- 当前方向展示层启用了独立二分类后处理，上限默认 `0.90`，超限会压缩并记录告警。
- 如果你希望更保守，可设置 `INDEXTRACK_DISPLAY_PROB_CAP=0.85`。

headline 趋势判定规则（展示层）：
- 若 `state=uncertain`，headline 直接显示“不确定”。
- 若 `signal_strength < 0.05`，headline 显示“不确定”。
- 若 `signal_strength < 0.15` 或 `top1-top2` 很小（默认 `<0.06`），headline 显示“偏上/偏下”。
- 仅当 `top1>=0.60` 且 `margin>=0.20` 且 `state!=uncertain` 时才显示“上行/下行”。

新模型样本不足或校准失败：
- 临时切换 `--model legacy` 验证数据链路是否正常。
- 或把周期改大（建议 `1Y` 或 `3Y`）增加训练样本。

报错 `OOF 样本不足，无法进行稳定校准`：
- 含义：`quantile` 模型训练窗口内可用样本太少，无法完成时间序列 OOF 校准。
- 现在 UI 在 `quantile` 模式会自动扩大训练窗口；若仍报错，通常是因为本次回退到了较短缓存快照。
- 解决：先保证网络可用，再运行一次 `--model quantile --period 1Y` 以刷新较长历史数据；或临时使用 `legacy`。

提示回退到缓存：
- 说明本次拉取实时数据失败，系统使用了本地最近一次成功数据。
- 可检查网络后再运行一次。

## 可选环境变量

- `INDEXTRACK_TIMEZONE` 默认时区
- `INDEXTRACK_DEFAULT_INDEX` 默认指数（`SP500/NASDAQ/BOTH`）
- `INDEXTRACK_DEFAULT_PERIOD` 默认周期（例如 `1Y`）
- `INDEXTRACK_REQUEST_TIMEOUT_SEC` 请求超时秒数
- `INDEXTRACK_PRIMARY_API_KEY` 预留字段（当前主源通常不依赖）
- `INDEXTRACK_MODEL` 默认模型（`quantile` / `legacy`）
- `INDEXTRACK_PROB_PROFILE` 概率模型参数档位（`optimized_v1` / `baseline`）

场景权重（可选）：
- `INDEXTRACK_SCENARIO_WEIGHTS_SHORT`
- `INDEXTRACK_SCENARIO_WEIGHTS_MID`
- `INDEXTRACK_SCENARIO_WEIGHTS_LONG`

概率模型参数（可选）：
- `INDEXTRACK_PROB_K_5 / INDEXTRACK_PROB_K_20 / INDEXTRACK_PROB_K_60`
- `INDEXTRACK_PROB_LAMBDA_5 / INDEXTRACK_PROB_LAMBDA_20 / INDEXTRACK_PROB_LAMBDA_60`
- `INDEXTRACK_PROB_SIGMA_FLOOR_5 / _20 / _60`
- `INDEXTRACK_PROB_CAP`
- `INDEXTRACK_DISPLAY_PROB_CAP`（方向展示层单边概率上限，默认 `0.90`）
- `INDEXTRACK_PROB_OOF_SPLITS`
- `INDEXTRACK_PROB_MIN_TRAIN_SIZE`
- `INDEXTRACK_PROB_MIN_VALID_SIZE`
- `INDEXTRACK_PROB_CALIB_MODE`（`softmax` / `conservative` / `none`）
- `INDEXTRACK_PROB_CALIB_MODE_5 / _20 / _60`（horizon 级覆盖）
- `INDEXTRACK_PROB_CALIB_TEMP`
- `INDEXTRACK_PROB_CALIB_BLEND_5 / _20 / _60`
- `INDEXTRACK_PROB_CALIB_MAX_SHIFT_5 / _20 / _60`

调试输出（`--prob-debug` + 概率细节）中可看到：
- `internal_state`（模型内部状态）
- `label`（方向标签：`up/down/uncertain`）
- `headline_label`（标题文案：`上行/偏上/不确定/偏下/下行`）
- `display_label`（最终展示标签编码）
- `top1` / `top2` / `margin`
- `signal_strength`（原 `confidence` 的展示层改名）
- `warning_level` / `warning_code` / `warning_message`

示例：
```bash
export INDEXTRACK_SCENARIO_WEIGHTS_SHORT="1.4,1.1,0.5,1.1,0.3"
```

quantile 参数档位 A/B 示例：
```bash
# 优化档（默认）
export INDEXTRACK_PROB_PROFILE=optimized_v1

# 回滚旧参数档
export INDEXTRACK_PROB_PROFILE=baseline
```

## Run Tests (English)
```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py" -v
```
