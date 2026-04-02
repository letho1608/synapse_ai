"""
Format kết quả tool sang thuần tiếng Việt trước khi inject vào prompt.
"""
from typing import Any, Dict

# Mô tả thời tiết tiếng Anh -> Việt
WEATHER_DESC_VI = {
    "clear": "Quang đãng",
    "sunny": "Nắng",
    "partly cloudy": "Nhiều mây",
    "partly cloud": "Nhiều mây",
    "cloudy": "Nhiều mây",
    "overcast": "U ám",
    "rain": "Mưa",
    "light rain": "Mưa nhẹ",
    "heavy rain": "Mưa to",
    "drizzle": "Mưa phùn",
    "thunderstorm": "Giông",
    "fog": "Sương mù",
    "mist": "Mù nhẹ",
    "snow": "Tuyết",
    "wind": "Có gió",
    "hot": "Nóng",
    "cold": "Lạnh",
}

# Lỗi máy tính tiếng Anh -> Việt
CALC_ERROR_VI = {
    "invalid syntax": "Biểu thức không hợp lệ",
    "division by zero": "Chia cho 0",
    "math domain error": "Lỗi miền toán học",
    "is not defined": "không được định nghĩa",
    "could not convert": "Không thể chuyển đổi",
}

# Mã ngôn ngữ -> tên tiếng Việt (dùng cho translate)
LANG_NAME_VI = {
    "en": "Tiếng Anh",
    "vi": "Tiếng Việt",
    "ja": "Tiếng Nhật",
    "ko": "Tiếng Hàn",
    "zh": "Tiếng Trung",
    "fr": "Tiếng Pháp",
    "de": "Tiếng Đức",
    "es": "Tiếng Tây Ban Nha",
    "ru": "Tiếng Nga",
    "th": "Tiếng Thái",
    "auto": "Tự động",
}


def _weather_vi(desc: str) -> str:
    if not desc:
        return "Không rõ"
    d = desc.strip().lower()
    for en, vi in WEATHER_DESC_VI.items():
        if en in d:
            return vi
    return desc  # giữ nguyên nếu không map được


def _calc_error_vi(err: str) -> str:
    if not err:
        return "Lỗi tính toán"
    e = err.strip().lower()
    for en, vi in CALC_ERROR_VI.items():
        if en in e:
            return vi
    return err


def _lang_vi(code: str) -> str:
    if not code:
        return code
    return LANG_NAME_VI.get(code.lower(), code)


def format_weather_vi(data: Dict[str, Any]) -> str:
    """Chuyển kết quả get_weather sang đoạn văn tiếng Việt."""
    if not data.get("success"):
        return f"Không lấy được thời tiết: {data.get('error', 'Lỗi')}."
    loc = data.get("location", "")
    temp = data.get("temperature", "N/A")
    feels = data.get("feels_like", "N/A")
    desc = data.get("description", "")
    humidity = data.get("humidity", "N/A")
    wind = data.get("wind_kmph", "N/A")
    unit = data.get("unit", "°C")
    desc_vi = _weather_vi(desc)
    return (
        f"Thời tiết tại {loc}: {desc_vi}, nhiệt độ {temp}{unit}, cảm giác như {feels}{unit}, "
        f"độ ẩm {humidity}%, gió {wind} km/h."
    )


def format_time_vi(data: Dict[str, Any]) -> str:
    """Chuyển kết quả get_current_time sang đoạn văn tiếng Việt."""
    if not data.get("success"):
        return f"Không lấy được thời gian: {data.get('error', 'Lỗi')}."
    dt = data.get("datetime", "")
    date = data.get("date", "")
    time = data.get("time", "")
    weekday = data.get("weekday_vi") or data.get("weekday", "")
    return f"Thời điểm hiện tại: {weekday}, ngày {date}, lúc {time} ({dt})."


def format_stock_vi(data: Dict[str, Any]) -> str:
    """Chuyển kết quả get_stock_price sang tiếng Việt."""
    if not data.get("success"):
        return f"Không lấy được giá: {data.get('error', 'Lỗi')}."
    sym = data.get("symbol", "")
    price = data.get("price", "N/A")
    change = data.get("change", 0)
    pct = data.get("change_percent", 0)
    cur = data.get("currency", "USD")
    up_down = "tăng" if change >= 0 else "giảm"
    return f"Giá cổ phiếu {sym}: {price} {cur}, {up_down} {abs(change)} ({pct}%)."


def format_exchange_vi(data: Dict[str, Any]) -> str:
    """Chuyển get_exchange_rate sang tiếng Việt."""
    if not data.get("success"):
        return f"Không lấy được tỷ giá: {data.get('error', 'Lỗi')}."
    f = data.get("from_currency", "")
    t = data.get("to_currency", "")
    rate = data.get("rate", "N/A")
    return f"Tỷ giá: 1 {f} = {rate} {t}."


def format_news_vi(data: Dict[str, Any]) -> str:
    """Chuyển get_news sang tiếng Việt."""
    if not data.get("success"):
        return f"Không lấy được tin: {data.get('error', 'Lỗi')}."
    articles = data.get("articles", [])[:5]
    lines = [f"- {a.get('title', '')}" for a in articles]
    return "Tin mới nhất:\n" + "\n".join(lines) if lines else "Không có tin."


def format_gold_vi(data: Dict[str, Any]) -> str:
    """Chuyển get_gold_price sang tiếng Việt."""
    if not data.get("success"):
        return f"Không lấy được giá vàng: {data.get('error', 'Lỗi')}."
    name = data.get("name", "Vàng")
    buy = data.get("buy", "N/A")
    sell = data.get("sell", "N/A")
    return f"Giá {name}: mua vào {buy} VND, bán ra {sell} VND."


def format_crypto_vi(data: Dict[str, Any]) -> str:
    """Chuyển get_crypto_price sang tiếng Việt."""
    if not data.get("success"):
        return f"Không lấy được giá: {data.get('error', 'Lỗi')}."
    sym = data.get("symbol", "")
    usd = data.get("usd", "N/A")
    vnd = data.get("vnd")
    ch = data.get("change_24h")
    vnd_str = f", khoảng {vnd} VND" if vnd else ""
    ch_str = f", thay đổi 24h: {ch}%" if ch is not None else ""
    return f"Giá {sym}: {usd} USD{vnd_str}{ch_str}."


def format_calculator_vi(data: Dict[str, Any]) -> str:
    """Chuyển kết quả calculate sang tiếng Việt (kèm map lỗi)."""
    if not data.get("success"):
        err = data.get("error", "Lỗi")
        return f"Tính toán thất bại: {_calc_error_vi(err)}."
    expr = data.get("expression", "")
    result = data.get("result", "N/A")
    return f"Kết quả: {expr} = {result}."


def format_translate_vi(data: Dict[str, Any]) -> str:
    """Chuyển kết quả translate sang tiếng Việt (nhãn + tên ngôn ngữ)."""
    if not data.get("success"):
        return f"Không dịch được: {data.get('error', 'Lỗi')}."
    orig = data.get("original", "")
    trans = data.get("translated", "")
    fl = _lang_vi(data.get("from_lang", ""))
    tl = _lang_vi(data.get("to_lang", ""))
    return f"Bản dịch ({fl} → {tl}): \"{trans}\""


def format_search_vi(data: Dict[str, Any]) -> str:
    """Chuyển kết quả search_web sang tiếng Việt (nhãn tiếng Việt)."""
    if not data.get("success"):
        return f"Không tìm được: {data.get('error', 'Lỗi')}."
    query = data.get("query", "")
    results = data.get("results", [])[:5]
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        snippet = (r.get("snippet", "") or "")[:200]
        link = r.get("link", "")
        lines.append(f"{i}. {title}\n   Đoạn trích: {snippet}...\n   Link: {link}")
    body = "\n".join(lines) if lines else "Không có kết quả."
    return f"Kết quả tìm kiếm cho \"{query}\":\n{body}"


def format_holidays_vi(data: Dict[str, Any]) -> str:
    """Chuyển get_holidays sang tiếng Việt (dữ liệu VN đã là tiếng Việt)."""
    if not data.get("success"):
        return f"Không kiểm tra được: {data.get('error', 'Lỗi')}."
    date = data.get("date", "")
    is_holiday = data.get("is_holiday", False)
    holidays = data.get("holidays", [])
    if is_holiday and holidays:
        return f"Ngày {date} (VN) là ngày lễ: {', '.join(holidays)}."
    return f"Ngày {date} (VN) không phải ngày lễ."


def format_tool_result_vi(tool_name: str, data: Dict[str, Any]) -> str:
    """Format kết quả một tool sang thuần tiếng Việt."""
    formatters = {
        "get_weather": format_weather_vi,
        "get_current_time": format_time_vi,
        "get_stock_price": format_stock_vi,
        "get_exchange_rate": format_exchange_vi,
        "get_news": format_news_vi,
        "get_gold_price": format_gold_vi,
        "get_crypto_price": format_crypto_vi,
        "calculate": format_calculator_vi,
        "translate": format_translate_vi,
        "search_web": format_search_vi,
        "get_holidays": format_holidays_vi,
    }
    fn = formatters.get(tool_name)
    if fn:
        return fn(data)
    # Fallback: mô tả chung
    if data.get("success") is False:
        return f"Lỗi: {data.get('error', 'Không xác định')}."
    import json
    return json.dumps(data, ensure_ascii=False)
