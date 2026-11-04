# ===========================================================================
# Gorzdrav Parser (requests + JSON API)
# ===========================================================================
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
BASE_URL = "https://gorzdrav.org"
SEARCH_API = BASE_URL + "/api/v1/product-search/ext"
CACHE_DIR = Path(".cache_gorzdrav")
SLEEP_BETWEEN = 0.3
MAX_WORKERS = 5
MAX_RETRIES = 3
RETRY_PAUSE = 0.5
PAGE_SIZE = 20

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Cache
# ---------------------------------------------------------------------------
_MISS = object()


def _cache_key(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def cache_get(key: str):
    p = CACHE_DIR / f"{_cache_key(key)}.json"
    if not p.exists():
        return _MISS
    return json.loads(p.read_text())


def cache_set(key: str, value):
    if value is None or value == []:
        return
    p = CACHE_DIR / f"{_cache_key(key)}.json"
    p.write_text(json.dumps(value, ensure_ascii=False))

# ---------------------------------------------------------------------------
# 3. Session
# ---------------------------------------------------------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        "Origin": BASE_URL,
        "Referer": BASE_URL + "/",
        "Flex-App": "WEB",
        "Flex-Region": "region=MOS",
        "Flex-Locale": "country=RU;bs=gz.ru",
    })
    return s

# ---------------------------------------------------------------------------
# 4. Clean drug name
# ---------------------------------------------------------------------------
def clean_drug_name(raw: str) -> str:
    s = re.sub(
        r"\b(таб\.?|табл\.?|капс\.?|кап\.?|крем|гель|мазь|спрей|капли|"
        r"р-р|раствор|сусп\.?|сироп|мг|мл|г|ме|iu|шт|№\s*\d+|\d+\s*шт)\b",
        " ", raw, flags=re.I,
    )
    s = re.sub(r"\d+([.,]\d+)?\s*%", " ", s)
    s = re.sub(r"\d+([.,]\d+)?", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" ,.-")
    return s

# ---------------------------------------------------------------------------
# 5. Parse one item
# ---------------------------------------------------------------------------
def _min_price(prices: list[dict]) -> float | None:
    values = [p.get("price") for p in (prices or []) if p.get("price") is not None]
    return min(values) if values else None


def _attr(attributes: list[dict], code: str) -> str | None:
    for a in (attributes or []):
        if a.get("code") == code:
            return a.get("value")
    return None


def parse_item(it: dict, drug_name: str) -> dict:
    prices = it.get("prices") or []
    customer = next((p for p in prices if p.get("type") == "customer"), None)
    price_regular = customer.get("price") if customer else None
    price_no_disc = customer.get("priceNoDiscount") if customer else None
    price_best = _min_price(prices)

    attrs = it.get("attributes") or []
    manufacturer = _attr(attrs, "manufacturer") or \
                   (it.get("additionalProperties") or {}).get("manufacturer")
    substance = _attr(attrs, "activeingredient")
    country = _attr(attrs, "country")

    return {
        "Поисковый запрос": drug_name,
        "Наименование": it.get("name"),
        "Цена (лучшая)": price_best,
        "Цена обычная": price_regular,
        "Цена без скидки": price_no_disc,
        "Действующее вещество": substance,
        "Производитель": manufacturer,
        "Страна": country,
        "Ссылка": f"{BASE_URL}/p/{it['url']}/" if it.get("url") else None,
    }

# ---------------------------------------------------------------------------
# 6. Fetch one drug
# ---------------------------------------------------------------------------
def fetch_drug(session: requests.Session, drug_name: str) -> list[dict]:
    search_term = clean_drug_name(drug_name) or drug_name
    cache_key = f"search:{search_term.lower()}"

    cached = cache_get(cache_key)
    if cached is not _MISS:
        return cached

    payload = {
        "page": 1,
        "size": 24,
        "sort": "POPULARITY",
        "desc": False,
        "filters": {
            "q": search_term,
        },
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(SEARCH_API, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            items = (((data.get("data") or {}).get("result") or {}).get("items")) or []
            products = [parse_item(it, drug_name) for it in items]

            cache_set(cache_key, products)
            return products
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                return []
            time.sleep(RETRY_PAUSE * attempt)
        except (ValueError, KeyError):
            return []

    return []

# ---------------------------------------------------------------------------
# 8. Pipeline: parsing
# ---------------------------------------------------------------------------
def parse_drugs(drug_names: list[str]) -> pd.DataFrame:
    session = make_session()
    all_products = []
    total = len(drug_names)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_drug = {
            executor.submit(fetch_drug, session, n): n
            for n in drug_names
        }
        for future in as_completed(future_to_drug):
            try:
                all_products.extend(future.result())
            except Exception:
                pass
            time.sleep(SLEEP_BETWEEN)

    df = pd.DataFrame(all_products)
    if not df.empty:
        df = df.drop_duplicates(subset=["Ссылка"]).reset_index(drop=True)

    log.info(
        "Parser Gorzdrav: готово | препаратов: %d | товаров: %d",
        total, len(df),
    )
    return df

# ---------------------------------------------------------------------------
# 10. Convenience wrapper
# ---------------------------------------------------------------------------
def run_parser(drug_names: list[str],
               output_path: Path = Path("gorzdrav_results.xlsx")) -> pd.DataFrame:
    return parse_drugs(drug_names)