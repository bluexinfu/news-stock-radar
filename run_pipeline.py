#!/usr/bin/env python3
"""
多主題批次執行管線（CLI 入口）

用法：
    python run_pipeline.py                         # 對所有主題跑完整管線
    python run_pipeline.py --topics cowos hbm      # 指定主題
    python run_pipeline.py --skip-collect          # 跳過採集，只做分析
    python run_pipeline.py --output my_report.html # 自訂輸出路徑

流程：
    1. 讀取 config/topics.yaml 的主題清單
    2. 對每個主題：
       a. 採集（prices + trends + news），若 --skip-collect 則跳過
       b. 對齊（alignment）
       c. 計算 NII（intensity）
    3. 用 theme_radar.rank_themes() 產生排行
    4. 生成「題材雷達」HTML 報告

⚠️ 注意：本腳本對外部 API（Google Trends / GDELT）有 rate limiting，
   首次對所有主題採集需要 10~30 分鐘，請耐心等待。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ── 輔助函式 ─────────────────────────────────────────────────────────

def load_topics_yaml() -> dict:
    cfg_path = ROOT / "config" / "topics.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_topic(topic: str, cfg: dict, skip_trends: bool = False) -> bool:
    """對單一主題跑採集。回傳 True=成功，False=部分失敗。"""
    start = cfg.get("time_range", {}).get("start", "2024-05-01")
    end   = cfg.get("time_range", {}).get("end",   date.today().strftime("%Y-%m-%d"))

    ok = True

    # 股價
    try:
        log.info("[%s] 採集股價…", topic)
        from src.collectors.prices import collect as collect_prices
        collect_prices(topic, start, end)
    except Exception as e:
        log.warning("[%s] 股價採集失敗：%s", topic, e)
        ok = False

    # Google Trends（CI 環境建議跳過，改由 weekly workflow 執行）
    if skip_trends:
        log.info("[%s] --skip-trends：跳過 Google Trends 採集（使用快取資料）", topic)
    else:
        try:
            log.info("[%s] 採集 Google Trends…", topic)
            from src.collectors.trends import collect as collect_trends
            collect_trends(topic, start, end)
        except Exception as e:
            log.warning("[%s] Trends 採集失敗：%s", topic, e)
            ok = False

    # Google News RSS + GDELT（統一用 collect() 入口，它會自行存檔）
    try:
        log.info("[%s] 採集 Google News RSS…", topic)
        from src.collectors.news import collect as collect_news
        collect_news(topic, start, end, source="googlenews")
    except Exception as e:
        log.warning("[%s] Google News 採集失敗：%s", topic, e)
        ok = False

    # GDELT（最慢，每次 6s，chunk 間 10s）
    try:
        log.info("[%s] 採集 GDELT（可能需要數分鐘）…", topic)
        from src.collectors.news import collect as collect_news_gdelt
        collect_news_gdelt(topic, start, end, source="gdelt")
    except Exception as e:
        log.warning("[%s] GDELT 採集失敗：%s", topic, e)
        ok = False

    return ok


def align_and_build_nii(topic: str) -> tuple | None:
    """對齊資料並計算 NII。回傳 (aligned_df, nii_df) 或 None。"""
    try:
        from src.processors.alignment import align
        aligned = align(topic)
        log.info("[%s] 對齊完成：%d 交易日 × %d 欄", topic, *aligned.shape)
    except Exception as e:
        log.error("[%s] 對齊失敗：%s", topic, e)
        return None

    try:
        from src.processors.intensity import build_nii_table
        nii_df = build_nii_table(topic)
        log.info("[%s] NII 計算完成：mean=%.2f  max=%.2f",
                 topic, nii_df["nii"].mean(), nii_df["nii"].max())
    except Exception as e:
        log.error("[%s] NII 計算失敗：%s", topic, e)
        return None

    return aligned, nii_df


# ── 主流程 ───────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    topics_cfg = load_topics_yaml()

    # 決定要跑哪些主題
    if args.topics and args.topics != ["all"]:
        selected = {k: v for k, v in topics_cfg.items() if k in args.topics}
        missing = set(args.topics) - set(selected)
        if missing:
            log.warning("找不到以下主題（跳過）：%s", missing)
    else:
        selected = topics_cfg

    if not selected:
        log.error("無可執行主題，結束")
        sys.exit(1)

    log.info("=== 管線開始：%d 個主題 %s ===", len(selected), list(selected))

    # ── Step 1：採集 ──────────────────────────────────────────────────
    if not args.skip_collect:
        for topic, cfg in selected.items():
            log.info("── 採集 [%s] ──", topic)
            collect_topic(topic, cfg, skip_trends=args.skip_trends)
            time.sleep(2)   # 主題切換間緩衝
    else:
        log.info("--skip-collect：跳過採集步驟")

    # ── Step 2：對齊 + NII ────────────────────────────────────────────
    topic_nii_map: dict[str, "pd.Series"] = {}
    topic_aligned_map: dict[str, "pd.DataFrame"] = {}
    display_names: dict[str, str] = {}

    for topic, cfg in selected.items():
        result = align_and_build_nii(topic)
        if result is None:
            log.warning("[%s] 跳過（對齊或 NII 失敗）", topic)
            continue
        aligned, nii_df = result
        topic_nii_map[topic]    = nii_df["nii"]
        topic_aligned_map[topic] = aligned
        display_names[topic]    = cfg.get("display_name", topic)

    if not topic_nii_map:
        log.error("沒有任何主題成功，結束")
        sys.exit(1)

    # ── Step 2.5：情緒分析（Phase 3-D）──────────────────────────────────
    from src.analyzers.sentiment import build_sentiment
    for topic in topic_nii_map:
        try:
            build_sentiment(topic, save=True)
        except Exception as e:
            log.debug("[%s] 情緒分析跳過：%s", topic, e)

    # ── Step 3：Theme Radar 排行 ──────────────────────────────────────
    from src.analyzers.theme_radar import rank_themes
    radar_df = rank_themes(topic_nii_map, display_names=display_names)

    log.info("\n=== 題材熱度排行 ===")
    for _, row in radar_df.iterrows():
        log.info(
            "#%d %s %s  NII=%.1f(z=%.2f)  7d斜率=%.3f  30d%%=%.1f%%  [%s]",
            row["rank"], row["phase_emoji"], row["display_name"],
            row["nii_latest"], row["nii_zscore"],
            row["nii_7d_slope"], row["nii_30d_pct_chg"],
            row["phase"],
        )

    # ── Step 4：SMI 建立 ──────────────────────────────────────────────
    from src.processors.sector_index import build_smi
    topic_smi_map: dict[str, "pd.Series"] = {}
    for topic, aligned in topic_aligned_map.items():
        try:
            smi = build_smi(topic, aligned, save=True)
            topic_smi_map[topic] = smi
        except Exception as e:
            log.warning("[%s] SMI 建立失敗：%s", topic, e)

    # ── Step 4.5：籌碼訊號（若有已採集的籌碼資料）────────────────────
    from src.processors.chips_signal import build_chips_signal
    topic_chips_map: dict[str, "pd.DataFrame"] = {}
    for topic in topic_nii_map:
        chips_path = ROOT / "data" / "raw" / "chips" / f"{topic}_institutional.parquet"
        if chips_path.exists():
            try:
                chips_df = build_chips_signal(topic, save=True)
                if not chips_df.empty:
                    topic_chips_map[topic] = chips_df
                    log.info("[%s] 籌碼訊號已加入", topic)
            except Exception as e:
                log.warning("[%s] 籌碼訊號計算失敗：%s", topic, e)
        else:
            log.debug("[%s] 無籌碼資料，跳過（可執行 src/collectors/chips.py 補全）", topic)

    # ── Step 5：生成 HTML 報告 ────────────────────────────────────────
    output = Path(args.output) if args.output else (
        ROOT / "reports" / f"radar_{date.today().strftime('%Y%m%d')}.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from src.radar_report_generator import generate_radar
        generate_radar(
            radar_df=radar_df,
            topic_nii_map=topic_nii_map,
            topic_smi_map=topic_smi_map,
            display_names=display_names,
            output_path=output,
            topic_chips_map=topic_chips_map if topic_chips_map else None,
        )
        log.info("✅ 報告已生成：%s", output)
    except Exception as e:
        log.error("報告生成失敗：%s", e, exc_info=True)
        sys.exit(1)

    # 同步報告到 HTTP Server 快取（/tmp/ns-reports）
    import shutil
    tmp_reports = Path("/tmp/ns-reports")
    try:
        tmp_reports.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, tmp_reports / output.name)
        log.info("📂 報告已同步到 http://localhost:8080/%s", output.name)
    except Exception as e:
        log.debug("同步 /tmp/ns-reports 失敗（不影響報告）：%s", e)

    # ── Step 6：資料品質治理（Phase 5-C）────────────────────────────
    if not args.skip_quality:
        try:
            import importlib.util
            qc_path = ROOT / "data" / "quality_check.py"
            spec = importlib.util.spec_from_file_location("quality_check", qc_path)
            qc = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(qc)
            log.info("── 資料品質檢查 ──")
            qc.run_check(list(selected.keys()))
        except Exception as e:
            log.warning("品質檢查失敗（不影響報告）：%s", e)

    # ── Step 7：Telegram 推播（Phase 5-A）────────────────────────────
    if not args.skip_alert:
        try:
            import importlib.util
            alert_path = ROOT / "scripts" / "daily_alert.py"
            spec = importlib.util.spec_from_file_location("daily_alert", alert_path)
            daily_alert = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(daily_alert)
            log.info("── 發送 Telegram 日報 ──")
            daily_alert.run(dry_run=False, force_send=False, report_filename=output.name)
        except Exception as e:
            log.warning("Telegram 推播失敗（不影響報告）：%s", e)


# ── CLI 入口 ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="多主題題材雷達管線",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--topics", nargs="+", default=["all"],
        help="要執行的主題（空白分隔）；預設 all = topics.yaml 中所有主題",
    )
    parser.add_argument(
        "--skip-collect", action="store_true",
        help="跳過資料採集，直接從已有資料做分析",
    )
    parser.add_argument(
        "--output", default=None,
        help="HTML 報告輸出路徑（預設：reports/radar_YYYYMMDD.html）",
    )
    parser.add_argument(
        "--skip-alert", action="store_true",
        help="跳過 Telegram 推播（只產生報告）",
    )
    parser.add_argument(
        "--skip-quality", action="store_true",
        help="跳過資料品質檢查",
    )
    parser.add_argument(
        "--skip-trends", action="store_true",
        help="跳過 Google Trends 採集（CI 環境建議使用，避免 429 限流）",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
