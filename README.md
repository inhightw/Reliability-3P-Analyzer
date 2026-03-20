# Reliability 3P Analyzer

這是一個專業級的可靠度 Web 分析工作站，專為處理帶有大量中斷測試 (Right-censored / Suspended) 的失效資料而設計。
本工具內建純數學最大概似估計 (MLE) 與商用軟體可靠度混合法 (Hybrid Method)，能精準估算 3 參數的 Weibull 與 Lognormal 分配參數，並利用 AIC/BIC 自動推薦最佳工程模型。

## 核心功能特色
- **純數學 MLE (Gamma >= 0)**：嚴格限制保證壽命大於零的演算法，能找出絕對機率最高的最大概似解。
- **Hybrid 混合演算法 (商規邏輯)**：運用 Kaplan-Meier 求取 Median Ranks 後搭配 Rank Regression 先求出最佳閾值 ($\gamma$)，有效避免 3 參數 Weibull 特有的「無限大崩潰陷阱 (Infinite Likelihood Trap)」。
- **自動決策引擎**：一鍵同時平行比對四種模型設定，以 AIC/BIC 作為複雜度懲罰基準，為工程師篩選出最合理、最符合物理現實的最佳配適模型。

## 安裝與執行方式

請先確保您的環境已經安裝 [uv](https://github.com/astral-sh/uv) 或 Python 原生環境，並執行以下指令：

```bash
# 1. 建立虛擬環境並啟動
uv venv .venv
# Windows PowerShell
.venv\Scripts\activate 
# macOS/Linux
# source .venv/bin/activate

# 2. 安裝套件
uv pip install -r requirements.txt

# 3. 啟動 Streamlit 網站
streamlit run app.py
```

## 關於演算法參考來源
- [Python `reliability` 套件說明：Fit_Lognormal_3P](https://reliability.readthedocs.io/en/latest/API/Fitters/Fit_Lognormal_3P.html#reliability.Fitters.Fit_Lognormal_3P)
- [ReliaSoft Weibull++ 官方白皮書與技術討論：Maximum Likelihood Estimation for the 3-Parameter Weibull Distribution](https://help.reliasoft.com/articles/content/hotwire/issue148/hottopics148.htm)
