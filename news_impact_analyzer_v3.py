#!/usr/bin/env python3
"""
predict_from_cnn_articles.py
────────────────────────────
Reads news stored in `cnn_articles.db` (table `articles`) and asks GPT‑4o
for one‑week directional predictions.

The script dumps the model reply into `./predictions/pred_YYYYMMDD_HHMM.json`.

Install deps:
    pip install --upgrade openai python-dateutil
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from dateutil import parser as dt
from openai import OpenAI

# ── 1. CONFIG ────────────────────────────────────────────────────────────────
from API import OPENAI_API_KEY  # ← your key lives here

BASE_DIR: Path = Path(__file__).resolve().parent
SRC_DB_PATH: str = str(BASE_DIR / "cnn_articles.db")  # source DB (news)
OUT_DIR: Path = BASE_DIR / "predictions"               # folder for JSON dumps

LOOKBACK_HOURS = 24          # how far back to collect articles
MODEL_NAME = "gpt-4o-mini"   # upgrade if you like
TEMPERATURE = 0.5           # keep it deterministic
MAX_ARTICLES_PER_TICKER = 8  # trim to stay under token limits

TICKERS: Dict[str, List[str]] = {
    # US large‑cap AI / infra
    "IBM":   ["IBM", "International Business Machines"],
    "MSFT":  ["Microsoft", "MS Corp", "MSFT"],
    "GOOGL": ["Google", "Alphabet", "GOOG"],
    "NVDA":  ["NVIDIA", "Nvidia"],
    "IONQ":  ["IonQ"],
    "RGTI":  ["Rigetti", "Rigetti Computing"],
    "QBTS":  ["D‑Wave", "DWave", "D‑Wave Quantum"],

    # European defence
    "BAESY":     ["BAE", "BAE Systems"],
    "RHM.DE":    ["Rheinmetall"],
    "LDO.MI":    ["Leonardo"],
    "HO.PA":     ["Thales"],
    "SAAB-B.ST": ["Saab"],
    "AM.PA":     ["Dassault", "Dassault Aviation"],
    "AIR.PA":    ["Airbus"],

    # U.S. AI stack
    "AMD":  ["AMD", "Advanced Micro Devices"],
    "META": ["Meta", "Facebook"],
    "AMZN": ["Amazon", "Amazon.com"],
    "PLTR": ["Palantir"],
    "AVGO": ["Broadcom"],
    "SNOW": ["Snowflake"],
    "SMCI": ["Super Micro", "Supermicro"],
    "INTC": ["Intel"],

    # Chinese majors
    "BABA":    ["Alibaba"],
    "BIDU":    ["Baidu"],
    "TCEHY":   ["Tencent"],
    "IFLYTEK": ["iFlytek", "IFLYTEK"],
    "0020.HK": ["SenseTime", "Sense Time"],

    # European AI / semis & software
    "ASML": ["ASML"],
    "SAP":  ["SAP"],
    "ARM":  ["Arm", "Arm Holdings"],
    "STM":  ["STMicro", "STMicroelectronics"],
    "DARK": ["Darktrace"],

    # Mega‑caps
    "AAPL": ["Apple"],
    "TSLA": ["Tesla"],
}

# ── 2. DATABASE → ARTICLES ───────────────────────────────────────────────────

def fetch_recent_articles(conn: sqlite3.Connection, since: datetime) -> List[Dict[str, Any]]:
    cur = conn.execute(
        """
        SELECT id, title, summary, published
          FROM articles
         WHERE published >= ?
        """,
        (since.isoformat(" "),),
    )
    return [
        {
            "id": row[0],
            "title": row[1] or "",
            "summary": row[2] or "",
            "published": dt.parse(row[3]),
        }
        for row in cur.fetchall()
    ]


# ── 3. OPENAI PROMPTING ──────────────────────────────────────────────────────
SYSTEM_MSG = """You are a financial news analyst.
For every ticker you receive, read the provided articles and decide whether the stock is likely to move UP, DOWN, or stay NEUTRAL in the next one‑week window. Base your call solely on the news supplied, taking NO OTHER DATA INTO ACCOUNT.
Return valid JSON in exactly this format:

{
  "TICKER": {"direction": "up|down|neutral", "confidence": 0.0-1.0, "reason": "..."},
  ...
}

Confidence should reflect the strength of the evidence. Keep each reason under 150 characters.
"""


def build_user_msg(ticker: str, articles: List[Dict[str, Any]]) -> str:
    if not articles:
        return f"Ticker: {ticker}\n\nNews:\n(no relevant articles)"

    lines = []
    for art in articles[:MAX_ARTICLES_PER_TICKER]:
        pub = art["published"].strftime("%Y-%m-%d %H:%M")
        lines.append(f"[{pub}] {art['title']} — {art['summary']}")
    return f"Ticker: {ticker}\n\nNews:\n" + "\n".join(lines)


def ask_gpt4o(client: OpenAI, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},  # ensures pure JSON
    )
    content = resp.choices[0].message.content
    return json.loads(content) if isinstance(content, str) else content


# ── 4. MAIN ──────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    client = OpenAI(api_key=OPENAI_API_KEY)
    since = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)

    with sqlite3.connect(SRC_DB_PATH) as conn:
        articles = fetch_recent_articles(conn, since)

    buckets: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TICKERS}
    for art in articles:
        text = f"{art['title']} {art['summary']}".lower()
        for ticker, aliases in TICKERS.items():
            if any(alias.lower() in text for alias in aliases):
                buckets[ticker].append(art)

    user_blocks = [build_user_msg(ticker, lst) for ticker, lst in buckets.items()]
    user_msg = "\n\n###\n\n".join(user_blocks)

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_msg},
    ]

    predictions = ask_gpt4o(client, messages)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path: Path = OUT_DIR / f"pred_{timestamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saved predictions → {out_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
