"""Giá vàng. Scrape hoặc API free."""
import aiohttp
import re
from typing import Any, Dict

from synapse.api.tools.cache import get_cached, set_cache


async def get_gold_price(unit: str = "SJC") -> Dict[str, Any]:
    """
    Lấy giá vàng (VN). unit: SJC, PNJ, 24K, 18K (tùy nguồn).
    Nguồn: có thể scrape từ sjc.com.vn hoặc dùng API công bố.
    """
    # Kiểm tra cache trước
    cached = get_cached("get_gold_price", unit=unit)
    if cached is not None:
        return cached
    
    try:
        # Một số trang công bố giá vàng (có thể thay bằng nguồn ổn định hơn)
        url = "https://www.sjc.com.vn/xml/tygiavang.xml"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return {"success": False, "error": f"Không lấy được dữ liệu (HTTP {resp.status})"}
                text = await resp.text()
        # Parse đơn giản XML hoặc regex
        buy = re.search(r"<buy>([0-9,]+)</buy>", text)
        sell = re.search(r"<sell>([0-9,]+)</sell>", text)
        name = re.search(r"<name>([^<]+)</name>", text)
        if buy and sell:
            result = {
                "success": True,
                "unit": unit,
                "name": name.group(1).strip() if name else "Vàng SJC",
                "buy": buy.group(1).replace(",", ""),
                "sell": sell.group(1).replace(",", ""),
                "currency": "VND",
            }
            # Lưu vào cache
            set_cache("get_gold_price", result, unit=unit)
            return result
        return {"success": False, "error": "Không parse được giá vàng từ nguồn"}
    except Exception as e:
        return {"success": False, "error": str(e)}
