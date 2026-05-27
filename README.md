# 訊息流 × 股價 對照分析系統

**專案代號**：`news-stock-correlation` | **Phase**：Phase 1 MVP | **版本**：v0.1

> 研究問題：關鍵零組件的訊息強度（新聞熱度 + 搜尋熱度），是否與相關概念股的股價存在可觀察的時序關係？訊息是領先、同步、還是落後股價？

---

## 快速開始

```bash
# 1. 建立虛擬環境
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 設定 API 金鑰（可選，P2 功能才需要）
cp .env.example .env
# 編輯 .env 填入 NewsAPI / SerpAPI 金鑰

# 4. 執行資料採集（先跑股價，最快）
python -m src.collectors.prices --topic cowos

# 5. 執行 Google Trends 採集（注意限流）
python -m src.collectors.trends --topic cowos

# 6. 執行新聞採集
python -m src.collectors.news --topic cowos

# 7. 開啟 Notebook 做 EDA
jupyter lab notebooks/01_data_exploration.ipynb
```

---

## 目錄結構

```
news-stock-correlation/
├── README.md
├── decisions.md           ← 方法論決策紀錄（必讀）
├── requirements.txt
├── .env.example
├── config/
│   └── topics.yaml        ← 主題/關鍵字/股票設定（換主題只改這裡）
├── data/
│   ├── raw/               ← API 抓回的原始資料
│   │   ├── trends/
│   │   ├── news/
│   │   └── prices/
│   └── processed/         ← 對齊、正規化後的時序資料（parquet）
├── src/
│   ├── collectors/        ← 資料採集
│   │   ├── trends.py      # Google Trends
│   │   ├── news.py        # Google News RSS + GDELT
│   │   └── prices.py      # yfinance / twstock
│   ├── processors/        ← 資料處理
│   │   ├── intensity.py   # NII 訊息強度指數計算
│   │   └── alignment.py   # 時序對齊
│   ├── analyzers/         ← 分析
│   │   ├── correlation.py # 皮爾森、滾動相關
│   │   ├── lead_lag.py    # 領先/落後分析
│   │   └── event_study.py # 事件研究
│   └── visualizers/
│       └── plots.py       ← 所有視覺化函式
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_phase1_cowos_analysis.ipynb
│   └── 03_robustness_check.ipynb
└── reports/
    └── phase1_cowos_report.html   ← 最終輸出報告
```

---

## 如何換主題

1. 在 `config/topics.yaml` 新增或選擇一個主題（目前有 `cowos`、`hbm`、`silicon_photonics`）
2. 執行採集時加上 `--topic <主題名稱>` 參數
3. 相關輸出會自動存到對應子目錄

---

## 訊息強度指數（NII）定義

```
NII_t = 0.5 × Trends_norm(t) + 0.5 × NewsCount_zscore(t)
```

- `Trends_norm`：Google Trends 原始值（0–100）
- `NewsCount_zscore`：當日新聞數量的 z-score
- 詳見 `decisions.md` D002

---

## 注意事項

- **弘塑科技代號待確認**：`topics.yaml` 中標記為 `VERIFY.TW`，請查詢正確代號後更新（見 `decisions.md` D001）
- **Google Trends 限流**：collector 已加 retry + 隨機 sleep；若仍失敗可人工匯出 CSV 放到 `data/raw/trends/`
- **相關不等於因果**：所有分析結果均為觀察性相關，不代表訊息驅動股價

---

## Phase 1 分析項目

| 項目 | 說明 | 優先級 |
|------|------|--------|
| A1 | 訊息強度與股價雙軸疊圖 | P0 |
| A2 | 全期間皮爾森相關係數 | P0 |
| A3 | 滾動相關係數（30/60 天窗口） | P0 |
| A4 | 領先/落後分析（lag = -10~+10 天） | P1 |
| A5 | 事件研究（訊息高峰日 ±10 天股價變化） | P1 |
| A6 | 多受惠股相關性比較熱圖 | P1 |
| A7 | 相對基準指數的超額相關 | P2 |
