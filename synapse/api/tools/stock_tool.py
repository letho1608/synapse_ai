"""Giá cổ phiếu. Dùng yfinance (scrape Yahoo Finance, không cần API key)."""
from typing import Any, Dict

from synapse.api.tools.cache import get_cached, set_cache

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


async def get_stock_price(symbol: str, exchange: str = None) -> Dict[str, Any]:
    """
    Lấy giá cổ phiếu hiện tại (hoặc phiên gần nhất).
    symbol: mã cổ phiếu (ví dụ AAPL, VNM, VIC).
    exchange: tùy chọn, ví dụ HOSE, HNX (yfinance thường tự nhận).
    """
    if not HAS_YFINANCE:
        return {"success": False, "error": "Chưa cài thư viện yfinance. Chạy: pip install yfinance"}
    
    # Kiểm tra cache trước
    cached = get_cached("get_stock_price", symbol=symbol, exchange=exchange)
    if cached is not None:
        return cached
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="5d")
        if hist.empty:
            return {"success": False, "error": f"Không có dữ liệu cho mã {symbol}"}
        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) >= 2 else last
        change = float(last["Close"]) - float(prev["Close"])
        change_pct = (change / float(prev["Close"]) * 100) if prev["Close"] else 0
        result = {
            "success": True,
            "symbol": symbol,
            "price": round(float(last["Close"]), 2),
            "open": round(float(last["Open"]), 2),
            "high": round(float(last["High"]), 2),
            "low": round(float(last["Low"]), 2),
            "change": round(change, 2),
            "change_percent": round(change_pct, 2),
            "volume": int(last["Volume"]) if "Volume" in last else None,
            "currency": info.get("currency", "USD"),
        }
        # Lưu vào cache
        set_cache("get_stock_price", result, symbol=symbol, exchange=exchange)
        return result
    except Exception as e:
        return {"success": False, "error": str(e), "symbol": symbol}
