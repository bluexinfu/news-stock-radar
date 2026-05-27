"""
視覺化函式庫

顏色規範（跨圖一致）：
  訊息強度：藍色系 (#1565C0)
  股價：    橘紅色系（各標的固定色）
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

plt.rcParams["font.family"] = ["Arial Unicode MS", "PingFang TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

TICKER_COLORS = {
    "TWII":    "#546E7A",
    "2330.TW": "#E53935",
    "3413.TW": "#FB8C00",
    "3583.TW": "#43A047",
    "3680.TWO": "#8E24AA",
    "3131.TWO": "#00ACC1",
}
TICKER_NAMES = {
    "TWII":    "台灣加權",
    "2330.TW": "台積電",
    "3413.TW": "京鼎",
    "3583.TW": "辛耘",
    "3680.TWO": "家登",
    "3131.TWO": "弘塑",
}

NII_COLOR = "#1565C0"


def _fmt_xaxis(ax, interval_months: int = 2):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval_months))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)


def _caption(fig, text: str):
    fig.text(0.01, -0.02, text, fontsize=8, color="gray")


# ── A1：NII × 股價雙軸疊圖 ─────────────────────────────────────────────────────

def plot_nii_vs_price(
    nii: pd.Series,
    price: pd.Series,
    ticker: str,
    nii_label: str = "NII 訊息強度",
    save_path=None,
):
    name = TICKER_NAMES.get(ticker, ticker)
    color_price = TICKER_COLORS.get(ticker, "#E53935")

    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.fill_between(nii.index, nii, alpha=0.2, color=NII_COLOR)
    ax1.plot(nii.index, nii, color=NII_COLOR, linewidth=1.8, label=nii_label)
    ax1.set_ylabel("NII 訊息強度", color=NII_COLOR)
    ax1.tick_params(axis="y", labelcolor=NII_COLOR)

    ax2 = ax1.twinx()
    ax2.plot(price.index, price, color=color_price, linewidth=1.8, label=f"{name} 收盤價")
    ax2.set_ylabel(f"{name} 收盤價（元）", color=color_price)
    ax2.tick_params(axis="y", labelcolor=color_price)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    _fmt_xaxis(ax1)
    plt.title(f"CoWoS 訊息強度 × {name}（{ticker}）雙軸疊圖", fontsize=12)
    _caption(fig, "資料來源：Google Trends + Google News RSS + GDELT + Yahoo Finance｜注意：相關不等於因果")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ── A3：滾動相關係數時序圖 ─────────────────────────────────────────────────────

def plot_rolling_correlation(
    rolling_df: pd.DataFrame,
    ticker: str,
    save_path=None,
):
    name = TICKER_NAMES.get(ticker, ticker)
    color_30 = "#1565C0"
    color_60 = "#90CAF9"

    fig, ax = plt.subplots(figsize=(14, 4))

    for col in rolling_df.columns:
        days = col.split("_")[-1]
        color = color_30 if "30" in col else color_60
        ax.plot(rolling_df.index, rolling_df[col], label=f"滾動相關 {days}", color=color, linewidth=1.5)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(0.3, color="green", linewidth=0.6, linestyle=":", alpha=0.6, label="r=0.3 參考線")
    ax.axhline(-0.3, color="red", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_ylabel("滾動皮爾森相關係數 r")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=9)
    _fmt_xaxis(ax)
    plt.title(f"NII × {name}（{ticker}）滾動相關係數（日報酬率）", fontsize=12)
    _caption(fig, "以日報酬率計算；窗口長度分別為 30 天與 60 天")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ── A4：Lag-Correlation 圖 ────────────────────────────────────────────────────

def plot_lag_correlation(
    lag_df: pd.DataFrame,
    ticker: str,
    best_lag_info: dict | None = None,
    save_path=None,
):
    name = TICKER_NAMES.get(ticker, ticker)
    fig, ax = plt.subplots(figsize=(12, 4))

    colors = [NII_COLOR if r >= 0 else "#E53935" for r in lag_df["r"]]
    bars = ax.bar(lag_df["lag"], lag_df["r"], color=colors, alpha=0.7, width=0.7)

    # 顯著性標記
    for _, row in lag_df.iterrows():
        if pd.notna(row["p_value"]) and row["p_value"] < 0.05:
            ax.text(row["lag"], row["r"] + (0.01 if row["r"] >= 0 else -0.02),
                    "*", ha="center", fontsize=10, color="black")

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Lag（天）\n正值 = NII 領先股價；負值 = NII 落後股價")
    ax.set_ylabel("皮爾森相關係數 r")
    ax.set_xticks(lag_df["lag"])

    title = f"NII × {name}（{ticker}）— Lead-Lag 分析"
    if best_lag_info:
        interp = best_lag_info.get("interpretation", "")
        ci_l = best_lag_info.get("ci_lower", "")
        ci_u = best_lag_info.get("ci_upper", "")
        title += f"\n最佳 lag={best_lag_info['best_lag']} ({interp})，r={best_lag_info['r_at_best_lag']:.3f}，95%CI [{ci_l}, {ci_u}]"

    plt.title(title, fontsize=11)
    _caption(fig, "* p < 0.05；以日報酬率計算；bootstrap n=500")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ── A5：事件研究圖 ────────────────────────────────────────────────────────────

def plot_event_study(
    car_df: pd.DataFrame,
    event_days: pd.DatetimeIndex,
    ticker: str,
    save_path=None,
):
    name = TICKER_NAMES.get(ticker, ticker)
    fig, ax = plt.subplots(figsize=(12, 5))

    # 個別事件（淡色）
    for col in car_df.columns:
        if col in ("mean", "median"):
            continue
        ax.plot(car_df.index, car_df[col] * 100, color="gray", linewidth=0.8, alpha=0.3)

    # 平均線
    ax.plot(car_df.index, car_df["mean"] * 100, color=NII_COLOR, linewidth=2.5, label="平均 CAR")
    ax.fill_between(
        car_df.index,
        car_df["mean"] * 100 - car_df.drop(columns=["mean", "median"]).std(axis=1) * 100,
        car_df["mean"] * 100 + car_df.drop(columns=["mean", "median"]).std(axis=1) * 100,
        alpha=0.15, color=NII_COLOR, label="±1 標準差"
    )

    ax.axvline(0, color="red", linewidth=1.5, linestyle="--", alpha=0.8, label="事件日（NII 峰值）")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("相對事件日（天）")
    ax.set_ylabel("累積報酬率（%）")
    ax.legend(fontsize=9)

    n_events = len([c for c in car_df.columns if c not in ("mean", "median")])
    plt.title(
        f"事件研究：NII 峰值日前後 {len(car_df)//2} 天 × {name}（{ticker}）\n"
        f"共 {n_events} 個事件日（NII > mean+1.5σ）",
        fontsize=11
    )
    dates_str = ", ".join(d.strftime("%Y-%m-%d") for d in sorted(event_days)[:5])
    if len(event_days) > 5:
        dates_str += f" ...（共 {len(event_days)} 個）"
    _caption(fig, f"事件日：{dates_str}｜CAR：以事件日歸零的累積報酬率")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ── A6：多標的相關係數熱圖 ────────────────────────────────────────────────────

def plot_correlation_heatmap(
    pearson_table: pd.DataFrame,
    save_path=None,
):
    fig, ax = plt.subplots(figsize=(8, 4))

    pivot = pearson_table.set_index("ticker")[["r"]].T
    mask = pearson_table.set_index("ticker")[["significant"]].T.rename(index={"significant": "r"})

    sns.heatmap(
        pivot.astype(float),
        annot=True, fmt=".3f",
        cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 11},
    )

    # 在不顯著的格子上加斜線標記
    for i, (_, row) in enumerate(pearson_table.iterrows()):
        if not row["significant"]:
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, fill=False,
                                        hatch="////", edgecolor="gray", linewidth=0))

    ax.set_title("NII × 各標的 皮爾森相關係數（日報酬率）\n斜線 = p ≥ 0.05，不顯著", fontsize=12)
    ax.set_ylabel("")
    ax.set_yticklabels(["r"], rotation=0)
    _caption(fig, "資料來源：Google Trends + Google News + GDELT + Yahoo Finance")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
