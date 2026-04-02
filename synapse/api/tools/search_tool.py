"""Tìm kiếm web. DuckDuckGo (free, không cần key)."""
from typing import Any, Dict, List

from synapse.api.tools.cache import get_cached, set_cache

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False


async def search_web(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Tìm kiếm thông tin trên web.
    query: từ khóa tìm kiếm.
    num_results: số kết quả (mặc định 5).
    """
    if not HAS_DDGS:
        return {"success": False, "error": "Chưa cài duckduckgo-search. Chạy: pip install duckduckgo-search"}
    
    # Kiểm tra cache trước
    cached = get_cached("search_web", query=query, num_results=num_results)
    if cached is not None:
        return cached
    
    try:
        with DDGS() as ddgs:
            results: List[Dict] = []
            for r in ddgs.text(query, max_results=min(int(num_results), 10)):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "link": r.get("href", ""),
                })
        result = {"success": True, "query": query, "count": len(results), "results": results}
        # Lưu vào cache
        set_cache("search_web", result, query=query, num_results=num_results)
        return result
    except Exception as e:
        return {"success": False, "error": str(e), "query": query}
