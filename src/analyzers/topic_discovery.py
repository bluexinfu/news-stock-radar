"""
自動題材發現（Phase 3-C）

流程：
  1. 載入廣域中文財經新聞標題（來自 news_broad.py）
  2. 用 jieba 斷詞 + 過濾財經停用詞
  3. 用 BERTopic + 多語 Sentence Transformer 做主題聚類
  4. 輸出「候選題材清單」：每個 cluster = 一個潛在市場題材
  5. 同時計算每個題材的「時間熱度趨勢」（週頻文章數）

⚠️ 發現的題材僅供參考，需人工確認後再加入 topics.yaml 監控。

用法：
  from src.analyzers.topic_discovery import run_discovery
  result = run_discovery(days=90)     # 用最近 90 天廣域新聞
  result.show()                       # 印出候選題材清單
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

# ── 中文財經停用詞 ────────────────────────────────────────────────────

FIN_STOPWORDS = {
    # 通用功能詞
    "的","了","是","在","和","與","也","但","而","這","那","有","為","以",
    "由","對","等","或","其","所","到","及","被","將","已","可","都","很",
    "更","再","雖","若","因","只","又","時","後","前","上","下","中","外",
    # 財經媒體高頻詞（本身無題材意義）
    "台股","股市","股票","盤中","盤後","今日","昨日","本週","本月","本季",
    "個股","標的","漲幅","跌幅","成交量","交易","市場","行情","走勢","表現",
    "指數","加權","上市","上櫃","投資","分析","報告","預測","展望","預期",
    "法人","外資","投信","自營","買超","賣超","持股","布局","調節","增持",
    "元","億","萬","%","％","億元","萬元","百分","點","跌","漲","元以",
    "消息","新聞","報導","說明","表示","指出","認為","估計","預計","建議",
    "公司","企業","廠商","業者","供應","需求","生產","出貨","客戶","訂單",
    "第一","第二","第三","第四","季","月","年","Q1","Q2","Q3","Q4",
    "ETF","基金","債券","匯率","利率","升息","降息",
    # ── 市場行情/技術分析術語（不代表投資題材）──────────────────────
    "漲停","跌停","三大","萬張","主力","多頭","空頭","飆漲","飆升",
    "亮燈","輪動","類股","資金","回補","助攻","看好","看壞","看多","看空",
    "轉強","轉弱","底部","高點","低點","突破","跌破","撐住","回落",
    "創高","創低","創歷史","黑馬","強勢","弱勢",
    # ── 公司治理/行政詞（不代表投資題材）──────────────────────────
    "董事","股利","決議","年度","子公司","有限公司","股份","基準日","發放日",
    "公告","申報","說明會","法說","董事會","股東會","除息","除權",
    # ── 其他雜訊詞 ────────────────────────────────────────────────
    "概念","題材","題材點","一文","凌厲","全面","齊","同","爆量","縮量",
    # ── 媒體/平台名稱（不代表投資題材）──────────────────────────────
    "yahoo","yahoo新聞","yahoo股市","yahoo財經",
    "cmoney","cmoney投資網誌","股市爆料同學會",
    "cnyes","鉅亨","鉅亨網","中央社","cna",
    "sinotrade","永豐金","豐雲學堂",
    "moneydj","moneyudn","pchome","msn",
    "經濟日報","工商時報","自由時報","聯合報","聯合財經",
    "理財周刊","財訊","smart","money錢",
    "旺得富","風傳媒","鏡週刊","天下雜誌","商業周刊",
    "即時新聞","產業即時新聞","熱門股","公告",
    # ── 版型/欄目名（無題材意義）──────────────────────────────────
    "follow法人","台股逐洞賽","今日漲停股","今日漲停跌停股",
    "日報","週報","早報","晚報","盤前","盤後分析",
    "股民嘆","股民笑","林彥呈",
}


def _jieba_tokenize(text: str, stopwords: set[str]) -> list[str]:
    """jieba 斷詞並過濾停用詞與單字。"""
    try:
        import jieba
        tokens = jieba.lcut(text)
    except ImportError:
        # fallback: 按標點切割
        tokens = re.split(r"[\s，。！？、,.\-/\[\]【】()（）「」『』〔〕：:；;]+", text)

    return [
        t for t in tokens
        if len(t) >= 2
        and t not in stopwords
        and not re.fullmatch(r"[\d\s\W]+", t)   # 排除純數字/符號
    ]


# ── BERTopic 主題建模 ─────────────────────────────────────────────────

DEFAULT_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _preprocess_title(title: str) -> str:
    """
    用 jieba 斷詞並過濾停用詞，供 CountVectorizer 使用。
    BERTopic 的 embedding 仍使用原始標題（保留語意）；
    c-TF-IDF 關鍵詞提取使用斷詞後結果（過濾媒體名稱）。
    """
    tokens = _jieba_tokenize(title, FIN_STOPWORDS)
    return " ".join(tokens) if tokens else title


def _build_bertopic(
    docs: list[str],
    n_topics: int | str = "auto",
    embed_model: str = DEFAULT_EMBED_MODEL,
    min_topic_size: int = 8,
    seed: int = 42,
):
    """
    建立 BERTopic 模型。

    Parameters
    ----------
    docs          : list[str]  — 文章標題清單（原始，供 embedding）
    n_topics      : int|"auto" — 目標主題數；"auto" 讓 HDBSCAN 自動決定
    embed_model   : str        — SentenceTransformer 模型名稱
    min_topic_size: int        — 每個 cluster 最少文章數
    seed          : int        — 隨機種子（UMAP reproducibility）

    Returns
    -------
    (topic_model, topics, probs)
    """
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN

    log.info("載入 Sentence Transformer：%s", embed_model)
    embedding_model = SentenceTransformer(embed_model)

    umap_model  = UMAP(n_neighbors=10, n_components=5,
                       min_dist=0.0, metric="cosine", random_state=seed)
    hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size,
                             metric="euclidean",
                             cluster_selection_method="eom",
                             prediction_data=True)

    # CountVectorizer 使用 jieba 斷詞結果，過濾媒體/平台名稱
    # analyzer=lambda x: x.split() 因為傳入的已是空白分隔的 token 字串
    vectorizer = CountVectorizer(
        analyzer="word",
        tokenizer=lambda x: x.split(),
        min_df=2,
        max_df=0.85,
        token_pattern=None,
    )
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

    nr_topics = None if n_topics == "auto" else n_topics

    topic_model = BERTopic(
        embedding_model         = embedding_model,
        umap_model              = umap_model,
        hdbscan_model           = hdbscan_model,
        vectorizer_model        = vectorizer,
        ctfidf_model            = ctfidf,
        nr_topics               = nr_topics,
        calculate_probabilities = False,
        verbose                 = False,
    )

    # embedding 用原始標題（保留完整語意）
    # c-TF-IDF 用斷詞後文本（過濾停用詞）
    log.info("預處理標題（jieba 斷詞 + 停用詞過濾）…")
    processed_docs = [_preprocess_title(d) for d in docs]

    log.info("BERTopic 訓練中（%d 篇文章）…", len(docs))
    topics, probs = topic_model.fit_transform(docs, embeddings=None)

    # 用斷詞文本更新關鍵詞表示
    try:
        topic_model.update_topics(processed_docs, vectorizer_model=vectorizer,
                                   ctfidf_model=ctfidf)
    except Exception as e:
        log.warning("update_topics 失敗（使用原始關鍵詞）：%s", e)

    n_found = len(set(topics)) - (1 if -1 in topics else 0)
    log.info("發現 %d 個主題（雜訊 -1：%d 篇）", n_found, topics.count(-1))

    return topic_model, topics, probs


# ── 候選題材整理 ─────────────────────────────────────────────────────

@dataclass
class ThemeCandidate:
    topic_id:      int
    keywords:      list[str]          # BERTopic 提取的代表詞
    doc_count:     int                # 文章數
    sample_titles: list[str]          # 代表性標題（前 5 篇）
    weekly_trend:  pd.Series          # 每週文章數趨勢
    suggested_name: str = ""          # 自動命名建議（可人工改）


def _auto_name(keywords: list[str]) -> str:
    """用前 2-3 個代表詞自動命名主題。"""
    return " / ".join(keywords[:3])


# 平台/格式詞：若 top-3 關鍵詞都是這些，視為無意義 cluster
_PLATFORM_WORDS = {
    "yahoo","cmoney","cnyes","sinotrade","moneydj","cna","msn","pchome",
    "公告","即時","日報","週報","follow","tw","sinotrade","news",
    "股民","林彥呈","豐雲","理財","財訊",
}


def _is_meaningful(keywords: list[str], threshold: int = 3) -> bool:
    """
    若 top-8 關鍵詞中，有意義財經詞 ≥ threshold 個，視為有意義主題。
    排除條件：關鍵詞全是平台名/市場格式詞，無具體技術/公司/產業詞。
    """
    # 把 FIN_STOPWORDS 也視為低意義詞（已在斷詞時過濾，但偶爾漏網）
    all_low_value = _PLATFORM_WORDS | {
        "三大","萬張","漲停","跌停","主力","多頭","空頭","飆漲",
        "亮燈","輪動","類股","資金","回補","助攻","看好","題材點",
        "凌厲","一文","創歷史","黑馬","轉強","底部","突破",
        "董事","股利","決議","年度","子公司","股份","公告",
    }
    low_count = sum(1 for kw in keywords[:8] if kw.lower() in all_low_value)
    meaningful_count = len(keywords[:8]) - low_count
    return meaningful_count >= threshold


def extract_candidates(
    df: pd.DataFrame,
    topics: list[int],
    topic_model,
    min_docs: int = 8,
) -> list[ThemeCandidate]:
    """
    從 BERTopic 結果整理出候選題材清單。

    Parameters
    ----------
    df          : 廣域新聞 DataFrame（含 title, published 欄）
    topics      : BERTopic.fit_transform 回傳的 topic 標籤
    topic_model : BERTopic 模型
    min_docs    : 最少文章數（過濾太小的 cluster）

    Returns
    -------
    list[ThemeCandidate]，依文章數降序排列
    """
    df = df.copy()
    df["_topic"] = topics

    candidates = []
    topic_info = topic_model.get_topic_info()

    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue   # 雜訊群

        topic_docs = df[df["_topic"] == tid]
        if len(topic_docs) < min_docs:
            continue

        # 代表詞
        kw_tuples = topic_model.get_topic(tid) or []
        keywords  = [w for w, _ in kw_tuples[:8]]

        # 過濾被媒體/平台名稱主導的 cluster（無投資意義）
        if not _is_meaningful(keywords):
            log.debug("跳過無意義 cluster #%d：%s", tid, keywords[:5])
            continue

        # 週頻趨勢
        weekly = (
            topic_docs.set_index("published")["title"]
            .resample("W").count()
            .rename("count")
        )

        candidate = ThemeCandidate(
            topic_id      = tid,
            keywords      = keywords,
            doc_count     = len(topic_docs),
            sample_titles = topic_docs["title"].head(5).tolist(),
            weekly_trend  = weekly,
            suggested_name = _auto_name(keywords),
        )
        candidates.append(candidate)

    candidates.sort(key=lambda c: c.doc_count, reverse=True)
    return candidates


# ── 主執行函式 ───────────────────────────────────────────────────────

@dataclass
class DiscoveryResult:
    candidates:   list[ThemeCandidate]
    topic_model:  object               # BERTopic model（供進一步分析）
    df:           pd.DataFrame         # 帶 _topic 欄的原始資料
    n_docs:       int = 0
    n_topics:     int = 0

    def show(self, top_n: int = 15) -> None:
        """印出前 N 個候選題材摘要。"""
        print(f"\n{'='*60}")
        print(f"自動發現題材：{self.n_topics} 個（共 {self.n_docs} 篇文章）")
        print(f"{'='*60}")
        for i, c in enumerate(self.candidates[:top_n], 1):
            latest_wk = c.weekly_trend.iloc[-1] if len(c.weekly_trend) else 0
            trend = "↗" if len(c.weekly_trend) >= 2 and c.weekly_trend.iloc[-1] > c.weekly_trend.iloc[-2] else "→"
            print(f"\n#{i:02d}  [{c.suggested_name}]  {c.doc_count} 篇  近週 {latest_wk} 篇 {trend}")
            print(f"     關鍵詞：{', '.join(c.keywords[:6])}")
            print(f"     例：{c.sample_titles[0][:55]}")

    def to_yaml_candidates(self, top_n: int = 10) -> str:
        """
        產生可貼入 topics.yaml 的候選草稿（需人工修改後使用）。
        """
        lines = ["# ⚠️ 以下為 BERTopic 自動發現的候選題材草稿"]
        lines += ["# 請人工確認關鍵字與成份股後再加入 topics.yaml\n"]
        for c in self.candidates[:top_n]:
            safe_name = re.sub(r"[^a-z0-9_]", "_",
                               c.suggested_name.lower().replace(" ", "_").replace("/", "_"))
            safe_name = re.sub(r"_+", "_", safe_name).strip("_")
            lines += [
                f"# 自動發現 #{c.topic_id}  文章數={c.doc_count}",
                f"{safe_name or f'topic_{c.topic_id}'}:",
                f"  display_name: \"{c.suggested_name}\"  # ← 請改成有意義的中文名稱",
                f"  keywords:",
                f"    primary: [\"{c.keywords[0] if c.keywords else ''}\"]",
                f"    chinese: {c.keywords[:4]}",
                f"    english: []   # ← 請補充英文關鍵字",
                f"  related_stocks:",
                f"    primary: []   # ← 請填入相關概念股 ticker",
                f"    secondary: []",
                f"  time_range:",
                f"    start: \"2026-01-01\"",
                f"    end: \"2026-12-31\"",
                f"  benchmark: \"^TWII\"\n",
            ]
        return "\n".join(lines)


def run_discovery(
    days: int = 90,
    n_topics: int | str = "auto",
    embed_model: str = DEFAULT_EMBED_MODEL,
    min_topic_size: int = 8,
    recollect: bool = False,
) -> DiscoveryResult:
    """
    完整自動題材發現流程。

    Parameters
    ----------
    days           : 分析最近幾天的新聞（預設 90）
    n_topics       : 主題數；"auto" 讓模型自行決定
    embed_model    : SentenceTransformer 模型名稱
    min_topic_size : 每個 cluster 最少文章數
    recollect      : 是否強制重新採集廣域新聞

    Returns
    -------
    DiscoveryResult
    """
    from src.collectors.news_broad import collect_broad, load_broad

    # 1. 載入廣域新聞
    if recollect:
        df = collect_broad(days=days, save=True)
    else:
        df = load_broad(days=days)

    if df.empty or len(df) < 20:
        log.error("廣域新聞資料不足（%d 篇），請先執行 collect_broad()", len(df))
        return DiscoveryResult([], None, df, 0, 0)

    log.info("廣域新聞：%d 篇  日期：%s ~ %s",
             len(df), df["published"].min().date(), df["published"].max().date())

    # 2. 準備文件（標題作為輸入）
    docs = df["title"].fillna("").tolist()

    # 3. BERTopic
    topic_model, topics, _ = _build_bertopic(
        docs, n_topics=n_topics,
        embed_model=embed_model,
        min_topic_size=min_topic_size,
    )

    # 4. 整理候選題材
    candidates = extract_candidates(df, topics, topic_model, min_docs=min_topic_size)
    n_topics_found = len(candidates)

    return DiscoveryResult(
        candidates  = candidates,
        topic_model = topic_model,
        df          = df.assign(_topic=topics),
        n_docs      = len(df),
        n_topics    = n_topics_found,
    )
