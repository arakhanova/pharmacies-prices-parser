from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = "https://www.rigla.ru"
GRAPHQL_URL = BASE_URL + "/graphql"
CACHE_DIR = Path(".cache_rigla")
SLEEP_BETWEEN = 0.3
MAX_WORKERS = 5
MAX_RETRIES = 3
RETRY_PAUSE = 0.5

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

NOTEBOOK_DIR = Path(r"C:\Users\aarak\Projects\Парсер_Ригла")
OUT = NOTEBOOK_DIR / "rigla_results.xlsx"

# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    })
    return s

# ---------------------------------------------------------------------------
# Cache
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
    p = CACHE_DIR / f"{_cache_key(key)}.json"
    p.write_text(json.dumps(value, ensure_ascii=False))

# ---------------------------------------------------------------------------
# GraphQL search query
# ---------------------------------------------------------------------------
SEARCH_QUERY = """
query getProductBySearchPhrase($query: String, $pageSize: Int!, $currentPage: Int!){
    products: productsElastic(
        search: $query,
        pageSize: $pageSize,
        sort: {
                is_in_stock: DESC,
                is_x: ASC,
                score: DESC,
                presentation: DESC,
                region_product_qty: DESC,
                name: ASC, category_priority:DESC,
                search_weight: DESC,
                product_qty: DESC
            },
        currentPage: $currentPage,
        applyFilter: {}
    ) {
        searched_phrases
        total_count
        page_info {
            page_size
            current_page
        }
        items {
            id
            sku
            name
            url_key
            is_isg
            price_isg
            price {
                oldPrice { amount { value } }
                regularPrice { amount { value } }
            }
            manufacturer_id { label }
            is_in_stock
        }
    }
}
"""


def parse_products_json(data: dict, drug_name: str) -> list[dict]:
    products = (data.get("data") or {}).get("products") or {}
    items = products.get("items") or []
    out = []
    for it in items:
        regular = (((it.get("price") or {}).get("regularPrice") or {})
                   .get("amount") or {}).get("value")
        isg = it.get("price_isg")
        candidates = [p for p in (regular, isg) if p is not None]
        price = min(candidates) if candidates else None

        out.append({
            "Поисковый запрос": drug_name,
            "Наименование": it.get("name"),
            "Цена": price,
            "Цена без скидки": regular,
            "Цена по карте": isg,
            "Производитель": (it.get("manufacturer_id") or {}).get("label"),
            "В наличии": it.get("is_in_stock"),
            "Ссылка": f"{BASE_URL}/product/{it['url_key']}" if it.get("url_key") else None,
        })
    return out

# ---------------------------------------------------------------------------
# Fetch one drug (GraphQL, with cache + retries)
# ---------------------------------------------------------------------------
def fetch_drug(session: requests.Session, drug_name: str) -> list[dict]:
    cache_key = f"search:{drug_name.lower()}"
    cached = cache_get(cache_key)
    if cached is not _MISS:
        return cached

    payload = {
        "query": SEARCH_QUERY,
        "variables": {
            "query": drug_name,
            "pageSize": 20,
            "currentPage": 1,
        },
    }
    headers = {
        "Content-Type": "application/json",
        "X-APP": "WEB",
        "Origin": BASE_URL,
        "Referer": f"{BASE_URL}/search?q={quote_plus(drug_name)}",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(GRAPHQL_URL, json=payload, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if "errors" in data:
                return []

            products = parse_products_json(data, drug_name)
            cache_set(cache_key, products)
            return products
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                return []
            time.sleep(RETRY_PAUSE * attempt)

    return []

# ---------------------------------------------------------------------------
# Pipeline: parsing
# ---------------------------------------------------------------------------
def parse_drugs(drug_names: list[str]) -> pd.DataFrame:
    """Только собирает данные с сайта. Ничего не пишет на диск."""
    session = make_session()
    all_products = []

    total = len(drug_names)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_drug = {
            executor.submit(fetch_drug, session, name): name
            for name in drug_names
        }
        for future in as_completed(future_to_drug):
            try:
                products = future.result()
                all_products.extend(products)
            except Exception:
                pass
            time.sleep(SLEEP_BETWEEN)

    df = pd.DataFrame(all_products)
    if not df.empty:
        df = df.drop_duplicates(subset=["Ссылка"]).reset_index(drop=True)

    log.info(
        "Parser Rigla: готово | препаратов: %d | товаров: %d",
        total, len(df),
    )
    return df

# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def run_parser(drug_names: list[str],
               output_path: Path = OUT) -> pd.DataFrame:
    return parse_drugs(drug_names)