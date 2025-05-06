"""
cnn_news_scraper.py  (fresh‑news filter)
=======================================
* Robust date parsing: uses `published_parsed`, then `updated_parsed`, else
  falls back to the current UTC time.
* Drops headlines older than **24 hours** (configurable via `MAX_AGE_HOURS`).
* Ensures every stored record has a valid `published` ISO timestamp so your
  downstream analytics never see 2023‑dated rows again.
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List

import aiosqlite
import feedparser

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_LEVEL = "INFO"
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "cnn_articles.db")
POLL_INTERVAL = 60  # seconds
MAX_AGE_HOURS = 24  # ignore articles older than this

FEEDS: List[str] = [
    "http://rss.cnn.com/rss/cnn_topstories.rss",
    "http://rss.cnn.com/rss/cnn_world.rss",
    "http://rss.cnn.com/rss/cnn_us.rss",
    "http://rss.cnn.com/rss/cnn_allpolitics.rss",
    "http://rss.cnn.com/rss/cnn_tech.rss",
    "http://rss.cnn.com/rss/cnn_health.rss",
    "http://rss.cnn.com/rss/cnn_showbiz.rss",
]

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def article_id(link: str) -> str:
    return hashlib.sha256(link.encode()).hexdigest()


def parse_datetime(entry: dict) -> datetime:
    """Return a timezone‑aware datetime for the entry or now() if missing."""
    if ts := entry.get("published_parsed"):
        return datetime(*ts[:6], tzinfo=timezone.utc)
    if ts := entry.get("updated_parsed"):
        return datetime(*ts[:6], tzinfo=timezone.utc)
    return datetime.now(tz=timezone.utc)


async def init_db(db: aiosqlite.Connection) -> None:
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            link TEXT NOT NULL,
            published DATETIME NOT NULL,
            summary TEXT,
            feed TEXT NOT NULL
        )
        """
    )
    await db.commit()


async def store_entry(db: aiosqlite.Connection, feed_url: str, entry: dict) -> bool:
    aid = article_id(entry.link)
    cur = await db.execute("SELECT 1 FROM articles WHERE id = ?", (aid,))
    if await cur.fetchone():
        return False  # duplicate

    published_dt = parse_datetime(entry)
    # Skip if the item is older than MAX_AGE_HOURS
    if published_dt < datetime.now(tz=timezone.utc) - timedelta(hours=MAX_AGE_HOURS):
        return False

    await db.execute(
        """
        INSERT INTO articles (id, title, link, published, summary, feed)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            aid,
            entry.get("title", ""),
            entry.link,
            published_dt.isoformat(),
            entry.get("summary", ""),
            feed_url,
        ),
    )
    await db.commit()
    return True


async def fetch_feed(db: aiosqlite.Connection, feed_url: str) -> int:
    parsed = feedparser.parse(feed_url)
    return sum([await store_entry(db, feed_url, e) for e in parsed.entries])


async def poll_loop() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await init_db(db)
        while True:
            start = datetime.now(tz=timezone.utc)
            added = await asyncio.gather(*(fetch_feed(db, url) for url in FEEDS))
            logger.info("Cycle finished | new_articles=%d", sum(added))
            elapsed = (datetime.now(tz=timezone.utc) - start).total_seconds()
            await asyncio.sleep(max(0, POLL_INTERVAL - elapsed))


if __name__ == "__main__":
    try:
        asyncio.run(poll_loop())
    except KeyboardInterrupt:
        logger.info("Interrupted by user; exiting.")
