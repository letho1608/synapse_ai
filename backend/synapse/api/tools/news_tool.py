"""Tin tức mới từ RSS. Không cần API key."""
from typing import Any, Dict, List

from synapse.api.tools.cache import get_cached, set_cache

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

# Một số nguồn RSS phổ biến
DEFAULT_FEEDS = {
    "vnexpress": "https://vnexpress.net/rss/tin-moi-nhat.rss",
    "dantri": "https://dantri.com.vn/rss/tin-moi-nhat.rss",
    "vietnamnet": "https://vietnamnet.vn/rss/tin-moi-nhat.rss",
    "thanhnien": "https://thanhnien.vn/rss/tin-moi-nhat.rss",
}


async def get_news(topic: str = None, source: str = None, limit: int = 10) -> Dict[str, Any]:
    """
    Lấy tin tức mới nhất từ RSS.
    topic: chủ đề (tùy chọn, có thể dùng để filter tiêu đề).
    source: vnexpress, dantri, vietnamnet, thanhnien; để trống thì lấy nhiều nguồn.
    limit: số tin tối đa.
    """
    if not HAS_FEEDPARSER:
        return {"success": False, "error": "Chưa cài feedparser. Chạy: pip install feedparser"}
    
    # Kiểm tra cache trước
    cached = get_cached("get_news", topic=topic, source=source, limit=limit)
    if cached is not None:
        return cached
    
    try:
        feeds = [DEFAULT_FEEDS[source]] if source and source in DEFAULT_FEEDS else list(DEFAULT_FEEDS.values())
        all_entries: List[Dict] = []
        for feed_url in feeds[:3]:
            try:
                parsed = feedparser.parse(feed_url)
                for e in parsed.get("entries", [])[:limit]:
                    title = e.get("title", "")
                    link = e.get("link", "")
                    published = e.get("published", "")
                    if topic and topic.lower() not in title.lower():
                        continue
                    all_entries.append({"title": title, "link": link, "published": published})
            except Exception:
                continue
        all_entries = all_entries[: int(limit)]
        result = {
            "success": True,
            "count": len(all_entries),
            "articles": all_entries,
        }
        # Lưu vào cache
        set_cache("get_news", result, topic=topic, source=source, limit=limit)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}
