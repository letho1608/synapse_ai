"""Dịch thuật. Có thể dùng LibreTranslate public hoặc API free."""
import aiohttp
from typing import Any, Dict

# LibreTranslate public instance (có thể đổi sang self-hosted)
LIBRETRANSLATE_URL = "https://libretranslate.com"


async def translate(text: str, from_lang: str = "auto", to_lang: str = "en") -> Dict[str, Any]:
    """
    Dịch đoạn văn bản.
    from_lang: auto hoặc mã ngôn ngữ (vi, en, ...).
    to_lang: mã ngôn ngữ đích (en, vi, ja, ...).
    """
    try:
        payload = {"q": text[:5000], "source": from_lang, "target": to_lang, "format": "text"}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{LIBRETRANSLATE_URL}/translate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return {"success": False, "error": f"HTTP {resp.status}"}
                data = await resp.json()
        translated = data.get("translatedText", "")
        return {"success": True, "original": text[:200], "translated": translated, "from_lang": from_lang, "to_lang": to_lang}
    except Exception as e:
        return {"success": False, "error": str(e)}
