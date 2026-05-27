"""
NII 驅動策略回測（Phase 4-A）

策略規則：
  1. 訊號：detect_phase() 從「冷卻」→「預熱」時買入 SMI
  2. 出場：detect_phase() 進入「降溫」，或持有超過 max_hold_days，或 NII z-score < -0.5
  3. 部位：一次只持有一個訊號，下個訊號前已出場

回測指標：
  - 各次交易（Entry/Exit/持有天/報酬）
  - 累積報酬 vs 買入持有 (buy & hold)
  - Sharpe Ratio（年化，無風險利率 1.5%）
  - 最大回撤（Max Drawdown）

用法：
  from src.analyzers.backtest import run_backtest, BacktestResult
  result = run_backtest("cowos")
  result.summary()
  result.plot()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR   = ROOT / "reports"

RISK_FREE_RATE = 0.015   # 年化 1.5%（台灣短率近似值）


# ── 輔助函式 ─────────────────────────────────────────────────────────

def _max_drawdown(cum_ret: pd.Series) -> float:
    """計算最大回撤（負值）。"""
    roll_max = cum_ret.cummax()
    drawdown = cum_ret / roll_max - 1
    return float(drawdown.min())


def _sharpe(daily_ret: pd.Series, risk_free_annual: float = RISK_FREE_RATE) -> float:
    """年化 Sharpe Ratio。"""
    rf_daily = (1 + risk_free_annual) ** (1 / 252) - 1
    excess   = daily_ret - rf_daily
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))


# ── 回測資料載入 ─────────────────────────────────────────────────────

def _load_data(topic: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    載入 NII + SMI 資料。

    Returns
    -------
    (nii_df, smi_return)
    nii_df 含 nii 欄；smi_return 為每日報酬率 Series
    """
    nii_path = PROCESSED_DIR / f"{topic}_nii.parquet"
    smi_path = PROCESSED_DIR / f"{topic}_smi.parquet"

    if not nii_path.exists():
        raise FileNotFoundError(f"NII 資料不存在：{nii_path}（請先跑 run_pipeline.py）")
    if not smi_path.exists():
        raise FileNotFoundError(f"SMI 資料不存在：{smi_path}（請先跑 run_pipeline.py）")

    nii_df     = pd.read_parquet(nii_path)
    nii_df.index = pd.to_datetime(nii_df.index)

    smi_df     = pd.read_parquet(smi_path)
    smi_df.index = pd.to_datetime(smi_df.index)
    smi_return = smi_df["smi_return"] if "smi_return" in smi_df.columns else smi_df.iloc[:, 0]

    return nii_df, smi_return


# ── 訊號生成 ─────────────────────────────────────────────────────────

def _generate_signals(
    nii_df: pd.DataFrame,
    slope_window: int = 14,
    cool_to_warm_only: bool = True,
) -> pd.Series:
    """
    從 NII 生成交易訊號（相位轉換）。

    Parameters
    ----------
    cool_to_warm_only : bool
        True  → 只在「冷卻」→「預熱」時買入（更保守）
        False → 任何進入「預熱」時買入（包含「降溫」→「預熱」）

    Returns
    -------
    pd.Series[int]  index=date  values=1（買入訊號）/ 0（無訊號）
    """
    from src.analyzers.theme_radar import compute_nii_slope

    nii    = nii_df["nii"]
    slope  = compute_nii_slope(nii, window=slope_window)
    mu     = nii.mean()
    sigma  = nii.std()

    # 重建每日的 phase
    def _phase(i: int) -> str:
        n = nii.iloc[i]
        s = slope.iloc[i]
        if n >= mu + sigma:
            return "發燒"
        elif n >= mu and s < 0:
            return "降溫"
        elif n < mu and s > 0:
            return "預熱"
        else:
            return "冷卻"

    phases = [_phase(i) for i in range(len(nii))]
    phase_series = pd.Series(phases, index=nii.index)

    # 訊號：前一天是「冷卻」，今天是「預熱」
    signals = pd.Series(0, index=nii.index)
    for i in range(1, len(phase_series)):
        curr = phase_series.iloc[i]
        prev = phase_series.iloc[i - 1]
        if curr == "預熱":
            if cool_to_warm_only:
                if prev == "冷卻":
                    signals.iloc[i] = 1
            else:
                if prev != "預熱":
                    signals.iloc[i] = 1

    return signals, phase_series


# ── 回測執行 ─────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_date:  pd.Timestamp
    exit_date:   pd.Timestamp
    entry_price: float       # SMI 累積指數（基期=100）
    exit_price:  float
    hold_days:   int
    ret:         float       # 單次報酬（含手續費後）
    exit_reason: str         # "降溫" / "max_hold" / "stop_loss"


@dataclass
class BacktestResult:
    topic:       str
    trades:      list[Trade]
    equity:      pd.Series          # 策略每日淨值（從1起）
    bnh_equity:  pd.Series          # 買入持有每日淨值（從1起）
    phase_series: pd.Series         # 每日 phase 標籤
    nii:         pd.Series          # NII 時序
    params:      dict = field(default_factory=dict)

    # ── 摘要統計 ─────────────────────────────────────────────────────

    def summary(self) -> dict:
        if not self.trades:
            return {"n_trades": 0, "total_ret": 0.0, "sharpe": 0.0, "max_dd": 0.0}

        rets    = [t.ret for t in self.trades]
        tot_ret = float((self.equity.iloc[-1] - 1) * 100)   # %
        bnh_ret = float((self.bnh_equity.iloc[-1] - 1) * 100)

        # 策略日報酬
        daily_ret = self.equity.pct_change().fillna(0)

        stats = {
            "主題":        self.topic,
            "交易次數":    len(self.trades),
            "勝率":        f"{sum(1 for r in rets if r > 0) / len(rets) * 100:.1f}%",
            "平均報酬":    f"{np.mean(rets) * 100:+.2f}%",
            "平均持有天":  f"{np.mean([t.hold_days for t in self.trades]):.1f}天",
            "策略累積報酬": f"{tot_ret:+.1f}%",
            "買持累積報酬": f"{bnh_ret:+.1f}%",
            "超額報酬":    f"{tot_ret - bnh_ret:+.1f}%",
            "年化Sharpe":  f"{_sharpe(daily_ret):.2f}",
            "最大回撤":    f"{_max_drawdown(self.equity) * 100:.1f}%",
        }
        return stats

    def print_summary(self) -> None:
        stats = self.summary()
        print("\n" + "=" * 50)
        print(f"  回測結果：{self.topic}")
        print("=" * 50)
        for k, v in stats.items():
            print(f"  {k:<12} {v}")
        print()
        if self.trades:
            print("  前 10 次交易：")
            print(f"  {'進場日':12} {'出場日':12} {'天數':5} {'報酬':8} {'原因'}")
            for t in self.trades[:10]:
                print(f"  {str(t.entry_date.date()):12} {str(t.exit_date.date()):12} "
                      f"{t.hold_days:5} {t.ret*100:+6.1f}%  {t.exit_reason}")

    def plot(self, save_path: Path | None = None) -> None:
        """繪製策略淨值 vs 買入持有 + NII 走勢。"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import logging as _logging
            _logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)
            plt.rcParams["font.family"] = ["Arial Unicode MS", "PingFang TC", "sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
        except ImportError:
            log.warning("matplotlib 未安裝，略過圖表")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # 上圖：淨值曲線
        ax1.plot(self.equity.index,     self.equity,     label="NII策略", color="#1565C0", linewidth=1.8)
        ax1.plot(self.bnh_equity.index, self.bnh_equity, label="買入持有", color="#9E9E9E",
                 linewidth=1.2, linestyle="--", alpha=0.8)

        # 標記進出場
        for t in self.trades:
            ax1.axvline(t.entry_date, color="#43A047", alpha=0.3, linewidth=0.8)
            ax1.axvline(t.exit_date,  color="#E53935", alpha=0.3, linewidth=0.8)

        ax1.set_ylabel("淨值（=1 起）", fontsize=9)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"{self.topic} — NII 訊號回測", fontsize=11)

        # 下圖：NII + 區間標色
        nii = self.nii.dropna()
        ax2.fill_between(nii.index, nii, alpha=0.15, color="#1565C0")
        ax2.plot(nii.index, nii, color="#1565C0", linewidth=1.2)

        mu    = nii.mean()
        sigma = nii.std()
        ax2.axhline(mu,         color="gray",   linewidth=0.7, linestyle="--", label=f"均值 {mu:.1f}")
        ax2.axhline(mu + sigma, color="#C62828", linewidth=0.6, linestyle=":", alpha=0.8)

        # 標記預熱期
        warm_mask = self.phase_series == "預熱"
        ax2.fill_between(nii.index,
                         nii.reindex(nii.index).fillna(0),
                         where=warm_mask.reindex(nii.index, fill_value=False),
                         alpha=0.3, color="#FF8F00", label="預熱")

        ax2.set_ylabel("NII", fontsize=9)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        for ax in (ax1, ax2):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, fontsize=8)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info("回測圖已存：%s", save_path)
        else:
            plt.show()
        plt.close(fig)


# ── 主函式 ───────────────────────────────────────────────────────────

def run_backtest(
    topic: str,
    slope_window: int = 14,
    max_hold_days: int = 30,
    fee: float = 0.001,              # 單邊手續費 0.1%
    cool_to_warm_only: bool = True,
    stop_loss: float = -0.12,        # 停損 -12%（相對 SMI）
    save_chart: bool = True,
) -> BacktestResult:
    """
    執行 NII 相位轉換策略回測。

    Parameters
    ----------
    topic          : str  — topics.yaml 主題 key
    slope_window   : int  — NII 斜率計算視窗（天）
    max_hold_days  : int  — 最大持有天數（超過強制出場）
    fee            : float — 單邊手續費比例
    cool_to_warm_only : bool — True: 只在冷→熱轉換時買入
    stop_loss      : float — 停損比例（負值，如 -0.12 = -12%）
    save_chart     : bool — 是否存回測圖到 reports/

    Returns
    -------
    BacktestResult
    """
    nii_df, smi_return = _load_data(topic)

    # 對齊時間軸
    common_idx = nii_df.index.intersection(smi_return.index)
    nii_df     = nii_df.loc[common_idx]
    smi_return = smi_return.loc[common_idx]

    log.info("[%s] 回測期間：%s ~ %s（%d 天）",
             topic, common_idx.min().date(), common_idx.max().date(), len(common_idx))

    # 生成相位 + 訊號
    signals, phase_series = _generate_signals(nii_df, slope_window, cool_to_warm_only)

    # SMI 累積指數（基期=100）
    smi_level = (1 + smi_return.fillna(0)).cumprod() * 100

    # ── 回測迴圈 ──────────────────────────────────────────────────────
    trades     = []
    position   = 0          # 0=空倉  1=持有
    entry_date = None
    entry_price = None
    hold_days  = 0

    equity_vals = pd.Series(1.0, index=common_idx)
    cash        = 1.0   # 初始淨值

    for i, date in enumerate(common_idx):
        curr_smi = float(smi_level.iloc[i])
        curr_ret = float(smi_return.iloc[i]) if i > 0 else 0.0

        if position == 1:
            # 持有中：更新淨值
            equity_vals.iloc[i] = equity_vals.iloc[i - 1] * (1 + curr_ret)
            hold_days += 1

            # 計算目前持有報酬
            curr_ret_from_entry = (curr_smi / entry_price) - 1 - 2 * fee

            # 出場判斷
            exit_reason = None
            if phase_series.iloc[i] == "降溫":
                exit_reason = "降溫"
            elif hold_days >= max_hold_days:
                exit_reason = "max_hold"
            elif curr_ret_from_entry <= stop_loss:
                exit_reason = "stop_loss"

            if exit_reason:
                ret = curr_ret_from_entry
                trades.append(Trade(
                    entry_date  = entry_date,
                    exit_date   = date,
                    entry_price = entry_price,
                    exit_price  = curr_smi,
                    hold_days   = hold_days,
                    ret         = ret,
                    exit_reason = exit_reason,
                ))
                cash     = equity_vals.iloc[i]   # 更新現金
                position = 0
                hold_days = 0
                log.debug("出場 %s：%s  報酬=%.1f%%", date.date(), exit_reason, ret * 100)

        else:
            # 空倉中：淨值不變（持現金）
            equity_vals.iloc[i] = equity_vals.iloc[i - 1] if i > 0 else 1.0

            # 進場訊號
            if signals.iloc[i] == 1 and i < len(common_idx) - 1:
                # 下一天開盤進場（簡化：次日收盤價進場）
                entry_date  = common_idx[i + 1]
                entry_price = float(smi_level.iloc[i + 1]) * (1 + fee)
                position    = 1
                hold_days   = 0
                log.debug("進場 %s：SMI=%.2f", entry_date.date(), entry_price)

    # 若期末仍持倉，強制平倉
    if position == 1:
        last_smi = float(smi_level.iloc[-1])
        ret = (last_smi / entry_price) - 1 - 2 * fee
        trades.append(Trade(
            entry_date  = entry_date,
            exit_date   = common_idx[-1],
            entry_price = entry_price,
            exit_price  = last_smi,
            hold_days   = hold_days,
            ret         = ret,
            exit_reason = "期末強平",
        ))

    # 買入持有淨值
    bnh_equity = (1 + smi_return.fillna(0)).cumprod()
    bnh_equity = bnh_equity / bnh_equity.iloc[0]   # 基期=1

    result = BacktestResult(
        topic        = topic,
        trades       = trades,
        equity       = equity_vals,
        bnh_equity   = bnh_equity,
        phase_series = phase_series,
        nii          = nii_df["nii"],
        params       = {
            "slope_window":       slope_window,
            "max_hold_days":      max_hold_days,
            "fee":                fee,
            "cool_to_warm_only":  cool_to_warm_only,
            "stop_loss":          stop_loss,
        },
    )

    if save_chart:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        chart_path = REPORTS_DIR / f"backtest_{topic}.png"
        result.plot(save_path=chart_path)

    return result


# ── CLI 入口 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format="%(asctime)s %(levelname)s %(name)s — %(message)s",
                         datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description="NII 驅動策略回測")
    parser.add_argument("--topic",         default="cowos", help="topics.yaml 主題 key")
    parser.add_argument("--max-hold",      type=int,   default=30,   help="最大持有天數")
    parser.add_argument("--slope-window",  type=int,   default=14,   help="NII 斜率視窗")
    parser.add_argument("--stop-loss",     type=float, default=-0.12, help="停損比例（負值）")
    parser.add_argument("--all-signals",   action="store_true",       help="任何預熱都進場（非只冷→熱）")
    args = parser.parse_args()

    result = run_backtest(
        topic             = args.topic,
        slope_window      = args.slope_window,
        max_hold_days     = args.max_hold,
        cool_to_warm_only = not args.all_signals,
        stop_loss         = args.stop_loss,
        save_chart        = True,
    )
    result.print_summary()
