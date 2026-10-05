"""
data/news_fetcher.py — Financial news headline aggregator

Fetches news headlines from RSS feeds of major financial outlets, filters
for US market relevance, and provides a list of recent headlines for
sentiment analysis.

Sources:
  - MarketWatch (Top Stories)
  - CNBC (Economy)
  - Google News (US markets search)
"""
from __future__ import annotations

import calendar
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

try:
    import feedparser
    _HAS_FEEDPARSER = True
except ImportError:
    _HAS_FEEDPARSER = False

from core import config
from core.logger import get_logger

logger = get_logger("NewsFetcher")

# RSS feed URLs
_RSS_FEEDS = [
    {
        "name": "MarketWatch",
        "url": "https://feeds.marketwatch.com/marketwatch/topstories/",
    },
    {
        "name": "CNBC Economy",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    },
    {
        "name": "Google News Markets",
        "url": "https://news.google.com/rss/search?q=US+stock+market+economy&hl=en-US&gl=US&ceid=US:en",
    },
]

# Keywords to filter for US market relevance
_RELEVANCE_KEYWORDS = [
    "stock", "market", "wall street", "dow", "nasdaq", "s&p",
    "fed", "federal reserve", "interest rate", "inflation",
    "jobs", "employment", "gdp", "economy", "recession",
    "dollar", "treasury", "bond", "yield", "earnings",
    "trade", "tariff", "cpi", "ppi", "retail sales",
    "bullish", "bearish", "rally", "selloff", "crash",
    "us100", "us30", "nasdaq 100", "dow jones", "es", "nq", "us100cash", "us500cash",
]


@dataclass
class NewsItem:
    """Represents a single news headline."""
    title: str
    summary: str
    source: str
    published: Optional[datetime]
    url: str


class NewsFetcher:
    """
    Aggregates financial news headlines from multiple RSS sources.
    Caches results with a configurable TTL to avoid redundant network calls.
    """

    def __init__(self) -> None:
        self._headlines: List[NewsItem] = []
        self._last_fetch: float = 0.0
        self._fetch_interval: float = config.NEWS_REFRESH_SEC

    def refresh(self, force: bool = False) -> None:
        """Fetch news from all RSS sources if cache has expired."""
        now = time.time()
        if not force and (now - self._last_fetch) < self._fetch_interval:
            return

        all_items: List[NewsItem] = []
        for feed_cfg in _RSS_FEEDS:
            items = self._fetch_feed(feed_cfg["url"], feed_cfg["name"])
            all_items.extend(items)

        # Filter for relevance
        relevant = self._filter_relevant(all_items)

        # Sort by recency (newest first) and deduplicate
        relevant.sort(key=lambda x: x.published or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        seen_titles = set()
        deduped = []
        for item in relevant:
            # Simple dedup by normalized title
            key = item.title.lower().strip()[:60]
            if key not in seen_titles:
                seen_titles.add(key)
                deduped.append(item)

        self._headlines = deduped[:30]  # Keep top 30 most recent
        self._last_fetch = now

        if self._headlines:
            logger.info(
                f"📰 News refreshed: {len(self._headlines)} relevant headlines "
                f"from {len(_RSS_FEEDS)} sources"
            )

    def get_headlines(self, max_items: int = 10) -> List[NewsItem]:
        """Return the most recent relevant headlines."""
        self.refresh()
        return self._headlines[:max_items]

    def get_headline_texts(self, max_items: int = 10) -> List[str]:
        """Return just the title strings (for sentiment analysis)."""
        return [h.title for h in self.get_headlines(max_items)]

    # ─── RSS Parsing ────────────────────────────────────────────────
    def _fetch_feed(self, url: str, source_name: str) -> List[NewsItem]:
        """Parse a single RSS feed and return NewsItem list."""
        if not _HAS_FEEDPARSER:
            logger.warning(
                "feedparser not installed. Run: pip install feedparser"
            )
            return []

        try:
            feed = feedparser.parse(url)
            items = []
            for entry in feed.entries[:20]:  # Limit per source
                pub_date = self._parse_date(entry)
                items.append(NewsItem(
                    title=entry.get("title", "").strip(),
                    summary=entry.get("summary", "").strip()[:200],
                    source=source_name,
                    published=pub_date,
                    url=entry.get("link", ""),
                ))
            return items

        except Exception as exc:
            logger.debug(f"Error fetching {source_name}: {exc}")
            return []

    # ─── Filtering ──────────────────────────────────────────────────
    @staticmethod
    def _filter_relevant(items: List[NewsItem]) -> List[NewsItem]:
        """Keep only items that mention US market-related keywords."""
        relevant = []
        for item in items:
            text = f"{item.title} {item.summary}".lower()
            if any(kw in text for kw in _RELEVANCE_KEYWORDS):
                relevant.append(item)
        return relevant

    @staticmethod
    def _parse_date(entry) -> Optional[datetime]:
        """Extract publication datetime from an RSS entry."""
        for attr in ("published_parsed", "updated_parsed"):
            tp = getattr(entry, attr, None)
            if tp:
                try:
                    ts = calendar.timegm(tp)
                    return datetime.fromtimestamp(ts, tz=timezone.utc)
                except (ValueError, OverflowError):
                    pass
        return None
