"""
Извлечение характеристик из наименований лекарств.
Гибрид: regex (основной) + LLM (fallback для сложных случаев).
Схема выхода ОДНА для обоих экстракторов:
  {brand, form, form_raw, dose_value, dose_unit, pack_count, pack_unit,
   volume_ml, is_combination, confidence, extractor}
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional
import pandas as pd

# ===========================================================================
# Словари
# ===========================================================================

FORMS = {
    "фл.-кап.": "drops",
    "фл.-кап": "drops",
    "фл. кап.": "drops",
    "флакон-капельница": "drops",
    "кап.-фл.": "drops",
    "капли гл.": "drops",
    "капли глазные": "drops",
    "таб.п/о плен.": "tab",
    "таблетки покрытые пленочной оболочкой": "tab",
    "таб.п/о": "tab", "таб.": "tab", "табл.": "tab", "таблетки": "tab",
    "таблетки, покрытые оболочкой": "tab",
    "таб.п/п/о": "tab", "таб. пролонг.": "tab", "таб.ретард": "tab",
    "таб.шип.": "tab", "таб.дисперг.": "tab", "таб. раств.": "tab",
    "таб. жев.": "tab",
    "капс.": "caps", "капсулы": "caps", "капс.киш.раст.": "caps",
    "капс.пролонг.": "caps",
    "р-р": "solution", "раствор": "solution", "р-р д/ин.": "solution",
    "р-р для ин.": "solution", "р-р для инф.": "solution",
    "р-р для в/в введ.": "solution", "р-р для в/м введ.": "solution",
    "р-р д/наруж.прим.": "solution", "конц.": "solution", "лиоф.": "solution",
    "спрей": "spray", "спрей наз.": "spray", "аэр.": "spray", "аэрозоль": "spray",
    "крем": "cream", "мазь": "ointment", "гель": "gel",
    "паста": "paste", "линимент": "liniment",
    "капли": "drops", "капли гл.": "drops", "капли наз.": "drops",
    "капли ушные": "drops",
    "свечи": "supp", "супп.": "supp", "супп.ваг.": "supp", "супп.рект.": "supp",
    "сироп": "syrup", "сусп.": "susp", "суспензия": "susp",
    "пор.": "powder", "порошок": "powder", "пор. для приг.сусп.": "powder",
    "гранулы": "granules",
    "пластырь": "patch", "ттс": "patch", "драже": "dragee",
    "сбор": "herbal", "настойка": "tincture", "экстракт": "extract",
        "табл п/п/о": "tab",       # ← добавить
    "табл. п/п/о": "tab",      # ← добавить
    "табл п/о": "tab",         # ← добавить
    "табл. п/о": "tab",        # ← добавить
    "таблетки п/п/о": "tab"
}

DOSE_UNITS = [
    "мг/мл", "мкг/мл", "мг/доза", "мкг/доза", "ме/мл", "ме/доза",
    "мкг", "мг", "г", "мл", "ме", "iu", "%",
]

PACK_UNITS = ["шт", "таб", "капс", "амп", "доз", "фл", "пак", "саше"]

ALLOWED_FORMS = {
    "tab", "caps", "solution", "spray", "cream", "ointment", "drops",
    "supp", "syrup", "powder", "gel", "patch", "dragee", "herbal",
    "tincture", "extract", "other",
}

# ===========================================================================
# Regex-экстрактор
# ===========================================================================

_DOSE_RE = re.compile(
    r"(\d+[.,]?\d*)\s*(" + "|".join(re.escape(u) for u in DOSE_UNITS) + r")",
    re.IGNORECASE,
)
_PACK_RE = re.compile(
    r"(?:№|N|n)\s*(\d+)|(\d+)\s*(?:шт|таб|капс|амп|доз|фл|пак|саше)\b",
    re.IGNORECASE,
)
_VOLUME_RE = re.compile(r"(\d+[.,]?\d*)\s*мл\b", re.IGNORECASE)
_COMBINATION_RE = re.compile(r"\d+\s*(?:мг|мкг|г)\s*\+\s*\d+")


def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[^\w\s/.,%№+]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _match_form_original(name: str) -> tuple[Optional[str], Optional[str]]:
    """Ищет форму в ИСХОДНОЙ строке. Возвращает (form_raw, form_norm)."""
    if not isinstance(name, str):
        return None, None
    name_lower = name.lower().replace("ё", "е")
    for key in sorted(FORMS, key=len, reverse=True):
        key_lower = key.lower().replace("ё", "е")
        if key_lower in name_lower:
            pos = name_lower.find(key_lower)
            if pos >= 0:
                return name[pos:pos + len(key)], FORMS[key]
    return None, None


def _parse_float(s: str) -> float:
    return float(s.replace(",", "."))


def extract_regex(name: str) -> Optional[dict]:
    """Извлекает характеристики через regex. brand и form_raw — как в исходной строке."""
    if not isinstance(name, str) or not name.strip():
        return None

    s = _normalize_text(name)

    form_raw, form_norm = _match_form_original(name)
    if form_norm is None:
        return None

    dose_value, dose_unit = None, None
    m = _DOSE_RE.search(s)
    if m:
        dose_value = _parse_float(m.group(1))
        dose_unit = m.group(2).lower()

    pack_count, pack_unit = None, None
    m = _PACK_RE.search(s)
    if m:
        pack_count = int(m.group(1) or m.group(2))
        pack_unit = "шт"

    volume_ml = None
    m = _VOLUME_RE.search(s)
    if m:
        volume_ml = _parse_float(m.group(1))

    name_lower = name.lower().replace("ё", "е")
    form_raw_lower = form_raw.lower().replace("ё", "е")
    pos = name_lower.find(form_raw_lower)
    if pos < 0:
        return None
    brand = name[:pos].strip().strip(" ,.-()[]")
    if not brand or len(brand) < 2:
        return None

    is_combination = bool(_COMBINATION_RE.search(s))

    if dose_value is not None:
        variants = {str(dose_value),
                    str(int(dose_value)) if dose_value.is_integer() else None}
        variants.discard(None)
        if not any(v in name for v in variants):
            dose_value, dose_unit = None, None

    if pack_count is not None and str(pack_count) not in name:
        pack_count, pack_unit = None, None

    return {
        "brand": brand,
        "form": form_norm,
        "form_raw": form_raw,
        "dose_value": dose_value,
        "dose_unit": dose_unit,
        "pack_count": pack_count,
        "pack_unit": pack_unit,
        "volume_ml": volume_ml,
        "is_combination": is_combination,
        "confidence": 1.0,
        "extractor": "regex",
    }

# ===========================================================================
# LLM-экстрактор (fallback)
# ===========================================================================

EXTRACT_PROMPT = """Ты — экстрактор структуры названий лекарств.
Верни СТРОГО JSON по схеме, без markdown.

Название: "{name}"

Схема:
{{
  "brand": str,                 // бренд КАК В НАЗВАНИИ: "Авелокс", "9 месяцев"
  "form": str,                  // канонический: tab, caps, solution, spray, cream,
                                // ointment, drops, supp, syrup, powder, gel, patch,
                                // dragee, herbal, tincture, extract, other
  "form_raw": str,              // форма КАК В НАЗВАНИИ: "таблетки покрытые пленочной оболочкой"
  "dose_value": float | null,
  "dose_unit": str | null,      // мг, мкг, г, мл, ме, %, мг/мл, мкг/доза
  "pack_count": int | null,
  "pack_unit": str | null,      // "шт" для штук/таблеток/капсул; "доз"; "амп"
  "volume_ml": float | null,
  "is_combination": bool,
  "confidence": float
}}

Пример:
Вход: "Авелокс таблетки покрытые пленочной оболочкой 400мг №5"
Выход: {{
  "brand": "Авелокс",
  "form": "tab",
  "form_raw": "таблетки покрытые пленочной оболочкой",
  "dose_value": 400.0,
  "dose_unit": "мг",
  "pack_count": 5,
  "pack_unit": "шт",
  "volume_ml": null,
  "is_combination": false,
  "confidence": 1.0
}}

Правила:
- brand и form_raw — КАК В НАЗВАНИИ (не lower, не без пробелов).
- form — ТОЛЬКО из канонического списка.
- pack_unit — "шт", если это таблетки/капсулы/штуки.
- dose_value и pack_count должны встречаться в названии.
- Если чего-то нет — null. Не выдумывай.
- Отвечай только JSON.
"""


def validate_llm_output(e: dict, name: str) -> dict:
    """
    Приводит ответ LLM к ТОЙ ЖЕ схеме, что extract_regex.
    Возвращает dict ровно с полями:
      brand, form, form_raw, dose_value, dose_unit, pack_count, pack_unit,
      volume_ml, is_combination, confidence, extractor
    """
    e = dict(e or {})

    # --- 1. Маппинг альтернативных имён полей ---
    if "dose_value" not in e and "dose_quantity" in e:
        e["dose_value"] = e["dose_quantity"]
    if "dose_unit" not in e and "dose_quantity_unit" in e:
        e["dose_unit"] = e["dose_quantity_unit"]
    if "pack_count" not in e and "pack_quantity" in e:
        e["pack_count"] = e["pack_quantity"]
    if "pack_unit" not in e and "pack_quantity_unit" in e:
        e["pack_unit"] = e["pack_quantity_unit"]
    if "volume_ml" not in e and "pack_volume_ml" in e:
        e["volume_ml"] = e["pack_volume_ml"]

    # --- 2. form — канонический ---
    form = (e.get("form") or "").lower().strip()
    if form not in ALLOWED_FORMS:
        form = "other"

    # --- 3. form_raw — если нет, берём из исходной строки ---
    form_raw = e.get("form_raw")
    if not form_raw or not isinstance(form_raw, str):
        _, form_from_name = _match_form_original(name)
        if form_from_name is not None:
            name_lower = name.lower().replace("ё", "е")
            for key in sorted(FORMS, key=len, reverse=True):
                key_lower = key.lower().replace("ё", "е")
                if key_lower in name_lower:
                    pos = name_lower.find(key_lower)
                    form_raw = name[pos:pos + len(key)]
                    break
        if not form_raw:
            form_raw = name[:30]
    e["form_raw"] = form_raw

    # --- 4. brand — как в исходной строке ---
    brand = e.get("brand")
    if brand:
        brand = str(brand).strip().strip(" ,.-()[]")
        if len(brand) < 2:
            brand = None
    if not brand:
        form_raw_lower = (e["form_raw"] or "").lower()
        pos = name.lower().find(form_raw_lower) if form_raw_lower else -1
        if pos > 0:
            brand = name[:pos].strip().strip(" ,.-()[]")
    e["brand"] = brand

    # --- 5. Сверка формы с regex: если regex нашёл — доверяем ему ---
    _, form_from_regex = _match_form_original(name)
    if form_from_regex is not None and form != form_from_regex:
        form = form_from_regex
        e["confidence"] = float(e.get("confidence", 1.0)) * 0.7
    e["form"] = form

    # --- 6. dose_value / dose_unit ---
    dv = e.get("dose_value")
    if dv is not None:
        try:
            dv = float(dv)
        except (TypeError, ValueError):
            dv = None
    if dv is None:
        e["dose_value"] = None
        e["dose_unit"] = None
    else:
        e["dose_value"] = dv
        du = e.get("dose_unit")
        e["dose_unit"] = (str(du).lower().strip() if du else None)

    # --- 7. pack_count / pack_unit ---
    pc = e.get("pack_count")
    if pc is not None:
        try:
            pc = int(float(pc))
        except (TypeError, ValueError):
            pc = None
    if pc is None:
        e["pack_count"] = None
        e["pack_unit"] = None
    else:
        e["pack_count"] = pc
        pu = (e.get("pack_unit") or "шт").lower().strip()
        e["pack_unit"] = "шт" if pu in PACK_UNITS else pu

    # --- 8. volume_ml ---
    vm = e.get("volume_ml")
    if vm is not None:
        try:
            vm = float(vm)
        except (TypeError, ValueError):
            vm = None
    e["volume_ml"] = vm

    # --- 9. is_combination ---
    e["is_combination"] = bool(e.get("is_combination", False))

    # --- 10. confidence ---
    try:
        conf = float(e.get("confidence", 1.0))
    except (TypeError, ValueError):
        conf = 1.0

    # --- 11. Валидация: числа должны быть в исходной строке ---
    if e["dose_value"] is not None:
        v = e["dose_value"]
        variants = {str(v), str(int(v)) if v.is_integer() else None,
                    str(v).replace(".", ",")}
        variants.discard(None)
        if not any(x in name for x in variants):
            e["dose_value"] = None
            e["dose_unit"] = None
            conf *= 0.5

    if e["pack_count"] is not None and str(e["pack_count"]) not in name:
        e["pack_count"] = None
        e["pack_unit"] = None
        conf *= 0.5

    e["confidence"] = max(0.0, min(1.0, conf))

    # --- 12. Финальная схема: РОВНО поля regex ---
    return {
        "brand": e["brand"],
        "form": e["form"],
        "form_raw": e["form_raw"],
        "dose_value": e["dose_value"],
        "dose_unit": e["dose_unit"],
        "pack_count": e["pack_count"],
        "pack_unit": e["pack_unit"],
        "volume_ml": e["volume_ml"],
        "is_combination": e["is_combination"],
        "confidence": e["confidence"],
        "extractor": "llm",
    }


class LLMExtractor:
    """LLM-экстрактор с кэшем. Вызывается только для названий, где regex не справился."""

    def __init__(self, client, cache_dir: Path, model: str = "gpt-4o-mini"):
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model

    def _cache_path(self, name: str) -> Path:
        h = hashlib.md5(name.encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    def extract(self, name: str) -> Optional[dict]:
        p = self._cache_path(name)
        if p.exists():
            return json.loads(p.read_text())

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user",
                           "content": EXTRACT_PROMPT.format(name=name)}],
                max_tokens=300,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw,
                             flags=re.MULTILINE).strip()
            e_raw = json.loads(raw)
            e = validate_llm_output(e_raw, name)
            p.write_text(json.dumps(e, ensure_ascii=False))
            return e
        except Exception as ex:
            print(f"[LLM] {name[:50]}: {type(ex).__name__}: {ex}")
            return None

# ===========================================================================
# Оркестратор
# ===========================================================================

def extract(name: str, llm: Optional[LLMExtractor] = None,
            min_confidence: float = 0.7) -> dict:
    """
    1. regex
    2. если regex не справился (None) или confidence < min_confidence — LLM
    3. если LLM нет или не справился — минимальная структура
    """
    e = extract_regex(name)

    if e is None or e.get("confidence", 0) < min_confidence:
        if llm is not None:
            e_llm = llm.extract(name)
            if e_llm is not None:
                return e_llm
        if e is None:
            return {
                "brand": None, "form": None, "form_raw": None,
                "dose_value": None, "dose_unit": None,
                "pack_count": None, "pack_unit": None,
                "volume_ml": None, "is_combination": False,
                "confidence": 0.0, "extractor": "none",
            }

    return e

# ===========================================================================
# Нормализация
# ===========================================================================

MASS = {"мкг": (0.001, "мг"), "мг": (1, "мг"), "г": (1000, "мг")}
VOL  = {"мл": (1, "мл"), "л": (1000, "мл")}


def to_base_unit(value: float, unit: str) -> tuple[float, str]:
    u = (unit or "").lower().replace(" ", "")
    if u in MASS:
        f, b = MASS[u]
        return round(value * f, 6), b
    if u in VOL:
        f, b = VOL[u]
        return round(value * f, 6), b
    return value, u


def normalize_entity(e: dict) -> Optional[dict]:
    """Добавляет brand_norm и dose_norm/unit_norm. НЕ меняет brand и form_raw."""
    if e is None:
        return None
    e = dict(e)

    if e.get("dose_value") is not None and e.get("dose_unit") is not None:
        try:
            dn, un = to_base_unit(float(e["dose_value"]), e["dose_unit"])
            e["dose_norm"] = dn
            e["unit_norm"] = un
        except (TypeError, ValueError):
            e["dose_norm"] = None
            e["unit_norm"] = None
    else:
        e["dose_norm"] = None
        e["unit_norm"] = None

    if e.get("brand"):
        e["brand_norm"] = e["brand"].lower().replace(" ", "").replace("ё", "е")
    else:
        e["brand_norm"] = None

    return e


def build_match_key(e: dict) -> Optional[str]:
    if e is None or e.get("brand_norm") is None or e.get("form") is None:
        return None
    return "|".join([
        e["brand_norm"],
        e["form"],
        f'{e["dose_norm"]}{e["unit_norm"]}' if e["dose_norm"] is not None else "?",
        f'{e["pack_count"]}{e["pack_unit"]}' if e["pack_count"] is not None else "?",
        f'{e["volume_ml"]}ml' if e.get("volume_ml") is not None else "novol",
    ])

# ===========================================================================
# Batch
# ===========================================================================

def extract_batch(
    long_df: pd.DataFrame,
    llm: Optional[LLMExtractor] = None,
    name_col: str = "name",
    n_workers: int = 10,
) -> pd.DataFrame:
    """Для каждой строки long_df извлекает характеристики. Добавляет entity_* и entity_match_key."""
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed

    unique_names = long_df[name_col].dropna().unique().tolist()
    print(f"[extract_batch] уникальных названий: {len(unique_names)}")

    entities = {}
    llm_todo = []
    for n in unique_names:
        e = extract_regex(n)
        if e is None or e.get("confidence", 0) < 0.7:
            llm_todo.append(n)
        entities[n] = e

    n_regex = len(unique_names) - len(llm_todo)
    print(f"[extract_batch] regex: {n_regex} / {len(unique_names)} "
          f"({n_regex/len(unique_names):.1%}), LLM: {len(llm_todo)}")

    if llm_todo and llm is not None:
        print(f"[extract_batch] LLM для {len(llm_todo)} названий...")
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(llm.extract, n): n for n in llm_todo}
            done = 0
            for fut in as_completed(futures):
                n = futures[fut]
                try:
                    e = fut.result()
                    if e is not None:
                        entities[n] = e
                except Exception as ex_:
                    print(f"[LLM] {n[:40]}: {ex_}")
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{len(llm_todo)}")

    rows = []
    for n in unique_names:
        e = normalize_entity(entities.get(n))
        if e is None:
            e = {"brand_norm": None, "form": None, "dose_norm": None,
                 "unit_norm": None, "pack_count": None, "pack_unit": None,
                 "volume_ml": None}
        e["name"] = n
        e["match_key"] = build_match_key(e) if e.get("brand_norm") else None
        rows.append(e)

    ent_df = pd.DataFrame(rows)

    keep = ["name", "brand", "form", "form_raw", "dose_value", "dose_unit",
            "pack_count", "pack_unit", "volume_ml", "is_combination",
            "confidence", "extractor", "brand_norm", "dose_norm", "unit_norm",
            "match_key"]
    keep = [c for c in keep if c in ent_df.columns]
    ent_df = ent_df[keep].rename(columns={
        c: f"entity_{c}" for c in keep if c != "name"
    })
    return long_df.merge(ent_df, on="name", how="left")
