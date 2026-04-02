"""Giá crypto. CoinGecko API (free, không cần key)."""
import aiohttp
from typing import Any, Dict

from synapse.api.tools.cache import get_cached, set_cache
from synapse.api.tools.rate_limiter import wait_if_needed


async def get_crypto_price(symbol: str) -> Dict[str, Any]:
    """
    Lấy giá cryptocurrency hiện tại.
    symbol: BTC, ETH, USDT, VN...
    """
    # Kiểm tra cache trước
    cached = get_cached("get_crypto_price", symbol=symbol)
    if cached is not None:
        return cached
    
    try:
        # Rate limiting: CoinGecko cho phép 10-50 calls/minute
        await wait_if_needed("get_crypto_price", max_calls=30, window_seconds=60, min_interval=1.0)
        
        symbol = symbol.upper().strip()
        # CoinGecko ids thường dùng
        id_map = {"BTC": "bitcoin", "ETH": "ethereum", "USDT": "tether", "USDC": "usd-coin", "BNB": "binancecoin", "XRP": "ripple", "SOL": "solana", "DOGE": "dogecoin"}
        coin_id = id_map.get(symbol, symbol.lower())
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd,vnd&include_24hr_change=true"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 429:
                    return {"success": False, "error": "Rate limit exceeded. Vui lòng thử lại sau."}
                if resp.status != 200:
                    return {"success": False, "error": f"HTTP {resp.status}"}
                data = await resp.json()
        if coin_id not in data:
            return {"success": False, "error": f"Không tìm thấy mã {symbol}"}
        p = data[coin_id]
        result = {
            "success": True,
            "symbol": symbol,
            "usd": p.get("usd"),
            "vnd": p.get("vnd"),
            "change_24h": p.get("usd_24h_change"),
        }
        # Lưu vào cache
        set_cache("get_crypto_price", result, symbol=symbol)
        return result
    except Exception as e:
        return {"success": False, "error": str(e), "symbol": symbol}
