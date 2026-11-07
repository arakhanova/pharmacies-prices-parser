import sys
import os
from pathlib import Path
from datetime import date
from sqlalchemy import create_engine, text
import pandas as pd
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. Настройка путей (работает и локально, и в GitHub Actions)
sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 2. Импорт локальных модулей
import parser_Rigla
import parser_Gorzdrav
import extractor

# 3. Загрузка исходного списка
try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    PROJECT_DIR = Path.cwd()

df_in = pd.read_excel(PROJECT_DIR / "drugs.xlsx", header=None)
drugs = df_in[0].dropna().astype(str).str.strip().tolist()

# ==========================================
# Вспомогательные функции парсинга
# ==========================================
def run_in_chunks(tasks, drug_list, chunk_size=200, max_workers=3):
    all_drugs = None
    for name, (module, drug_list) in tasks.items():
        all_drugs = drug_list
        break
    chunks = [all_drugs[i:i+chunk_size] for i in range(0, len(all_drugs), chunk_size)]
    print(f"Чанков: {len(chunks)} × {chunk_size} препаратов")

    acc = {name: [] for name in tasks}
    t0 = time.time()

    for ci, chunk in enumerate(chunks, 1):
        print(f"\n=== Чанк {ci}/{len(chunks)} ({len(chunk)} препаратов) ===")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(module.parse_drugs, chunk): name
                for name, (module, drug_list) in tasks.items()
            }
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    df = fut.result()
                    if df is not None and not df.empty:
                        acc[name].append(df)
                    print(f"  [{name:10s}] {0 if df is None else df.shape[0]} строк")
                except Exception as e:
                    print(f"  [{name:10s}] ОШИБКА: {type(e).__name__}: {e}")

        import gc
        gc.collect()
        elapsed = time.time() - t0
        eta = elapsed / ci * (len(chunks) - ci)
        print(f"  прогресс: {elapsed:.0f}s, ETA {eta:.0f}s")

    results = {}
    for name, frames in acc.items():
        if frames:
            results[name] = pd.concat(frames, ignore_index=True)
            results[name] = results[name].drop_duplicates(subset=["Ссылка"]).reset_index(drop=True)
        else:
            results[name] = pd.DataFrame()
        print(f"[{name:10s}] итого: {results[name].shape}")
    return results

def normalize_pharmacy_data(df, pharmacy_name):
    SCHEMAS = {
        "gorzdrav": {
            "rename": {
                "Поисковый запрос": "query", "Наименование": "name",
                "Цена (лучшая)": "price", "Цена без скидки": "price_regular",
                "Производитель": "manufacturer", "Действующее вещество": "active_substance",
                "Страна": "country", "Ссылка": "url",
            },
        },
        "rigla": {
            "rename": {
                "Поисковый запрос": "query", "Наименование": "name",
                "Цена": "price", "Цена без скидки": "price_regular",
                "Цена по карте": "price_card", "Производитель": "manufacturer",
                "В наличии": "in_stock", "Ссылка": "url",
            },
        },
    }
    UNIFIED_COLS = [
        "pharmacy", "query", "name", "price", "price_regular", "price_card",
        "manufacturer", "active_substance", "country", "in_stock", "url",
    ]

    if df is None or df.empty:
        print(f"Данные для '{pharmacy_name}' пусты.")
        return pd.DataFrame(columns=UNIFIED_COLS)

    rename_map = SCHEMAS.get(pharmacy_name, {}).get("rename", {})
    df = df.rename(columns=rename_map).copy()

    for col in UNIFIED_COLS:
        if col not in df.columns:
            df[col] = None

    df["pharmacy"] = pharmacy_name
    df = df[UNIFIED_COLS]

    for col in ["price", "price_regular", "price_card"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ==========================================
# Многоступенчатый матчинг
# ==========================================
def build_pharmacy_summary(df, pharmacy_name, drugs_list):
    pharmacy_raw = df[
        (df["pharmacy"] == pharmacy_name) & 
        (df["query"].isin(drugs_list))
    ].copy()
    
    if pharmacy_raw.empty:
        print(f"⚠️ Для аптеки '{pharmacy_name}' не найдено данных.")
        return pd.DataFrame({
            "Исходное наим товара": drugs_list,
            f"Товар {pharmacy_name.capitalize()}": None,
            f"Цена {pharmacy_name.capitalize()}": None,
            f"Ссылка {pharmacy_name.capitalize()}": None
        })

    def get_entity_data(x):
        e = extractor.extract_regex(x)
        if not e:
            return {"key": None, "brand": None, "dose": None}
        norm = extractor.normalize_entity(e)
        return {
            "key": extractor.build_match_key(norm),
            "brand": norm.get("brand_norm"),
            "dose": norm.get("dose_norm")
        }

    # Считаем один раз для выдачи и для списка
    pharm_ents = pharmacy_raw["name"].apply(get_entity_data)
    pharmacy_raw["parsed_match_key"] = pharm_ents.apply(lambda x: x["key"])
    pharmacy_raw["parsed_brand"] = pharm_ents.apply(lambda x: x["brand"])
    pharmacy_raw["parsed_dose"] = pharm_ents.apply(lambda x: x["dose"])

    drugs_df = pd.DataFrame({"Исходное наим товара": drugs_list})
    target_ents = drugs_df["Исходное наим товара"].apply(get_entity_data)
    drugs_df["target_match_key"] = target_ents.apply(lambda x: x["key"])
    drugs_df["target_brand"] = target_ents.apply(lambda x: x["brand"])
    drugs_df["target_dose"] = target_ents.apply(lambda x: x["dose"])

    # Ступень 1: Строгий
    print(f"[{pharmacy_name.capitalize()}] Ступень 1: Строгий матчинг...")
    matched = pd.merge(
        drugs_df, pharmacy_raw,
        left_on=["Исходное наим товара", "target_match_key"],
        right_on=["query", "parsed_match_key"],
        how="left"
    )

    # Ступень 2: Бренд + Доза
    missing_mask = matched["name"].isna()
    if missing_mask.any():
        print(f"[{pharmacy_name.capitalize()}] Ступень 2: Матчинг по бренду и дозе ({missing_mask.sum()} записей)...")
        valid_pharm = pharmacy_raw.dropna(subset=["parsed_brand", "parsed_dose", "price"])
        best_by_brand_dose = valid_pharm.sort_values("price").groupby(["parsed_brand", "parsed_dose"]).first().reset_index()
        
        def rescue_by_brand_dose(row):
            b, d = row["target_brand"], row["target_dose"]
            if pd.isna(b) or pd.isna(d):
                return pd.Series({"name": None, "price": None, "url": None})
            match = best_by_brand_dose[(best_by_brand_dose["parsed_brand"] == b) & (best_by_brand_dose["parsed_dose"] == d)]
            if not match.empty:
                best = match.iloc[0]
                return pd.Series({"name": best["name"], "price": best["price"], "url": best["url"]})
            return pd.Series({"name": None, "price": None, "url": None})

        rescued_step2 = matched[missing_mask].apply(rescue_by_brand_dose, axis=1)
        matched.loc[missing_mask, ["name", "price", "url"]] = rescued_step2[["name", "price", "url"]].values

    # Ступень 3: Только бренд
    missing_mask = matched["name"].isna()
    if missing_mask.any():
        print(f"[{pharmacy_name.capitalize()}] Ступень 3: Матчинг по бренду ({missing_mask.sum()} записей)...")
        valid_pharm_brand = pharmacy_raw.dropna(subset=["parsed_brand", "price"])
        best_by_brand = valid_pharm_brand.sort_values("price").groupby("parsed_brand").first().reset_index()
        
        def rescue_by_brand_only(row):
            b = row["target_brand"]
            if pd.isna(b):
                return pd.Series({"name": None, "price": None, "url": None})
            match = best_by_brand[best_by_brand["parsed_brand"] == b]
            if not match.empty:
                best = match.iloc[0]
                return pd.Series({"name": best["name"], "price": best["price"], "url": best["url"]})
            return pd.Series({"name": None, "price": None, "url": None})

        rescued_step3 = matched[missing_mask].apply(rescue_by_brand_only, axis=1)
        matched.loc[missing_mask, ["name", "price", "url"]] = rescued_step3[["name", "price", "url"]].values

    # Ступень 4: Поиск по первому слову (ОПТИМИЗИРОВАНО)
    missing_mask = matched["name"].isna()
    if missing_mask.any():
        print(f"[{pharmacy_name.capitalize()}] Ступень 4: Поиск по первому слову ({missing_mask.sum()} записей)...")
        
        missing_queries = matched.loc[missing_mask, "Исходное наим товара"].tolist()
        first_words = [re.sub(r"[^а-яa-z]", "", q.lower().split()[0]) for q in missing_queries]
        pharm_names_lower = pharmacy_raw["name"].str.lower().fillna("")
        unique_words = set(w for w in first_words if w)
        
        best_items = {}
        for word in unique_words:
            mask = pharm_names_lower.str.contains(word, na=False)
            candidates = pharmacy_raw[mask]
            if not candidates.empty:
                best_items[word] = candidates.sort_values("price").iloc[0]

        for idx in matched[missing_mask].index:
            query = matched.loc[idx, "Исходное наим товара"]
            word = re.sub(r"[^а-яa-z]", "", query.lower().split()[0])
            if word in best_items:
                best = best_items[word]
                matched.loc[idx, "name"] = best["name"]
                matched.loc[idx, "price"] = best["price"]
                matched.loc[idx, "url"] = best["url"]

    # Финальная агрегация
    matched_data = (
        matched
        .groupby("Исходное наим товара")
        .agg({"name": "first", "price": "first", "url": "first"})
        .reset_index()
    )

    matched_data = pd.merge(
        pd.DataFrame({"Исходное наим товара": drugs_list}),
        matched_data, on="Исходное наим товара", how="left"
    )

    cap_name = pharmacy_name.capitalize()
    matched_data.columns = ["Исходное наим товара", f"Товар {cap_name}", f"Цена {cap_name}", f"Ссылка {cap_name}"]

    found = matched_data[f"Товар {cap_name}"].notna().sum()
    print(f"✅ [{cap_name}] Найдено: {found} из {len(drugs_list)} ({found/len(drugs_list):.1%})")

    return matched_data

# ==========================================
# Запуск парсинга и сборка
# ==========================================
tasks = {
    "gorzdrav":  (parser_Gorzdrav,  drugs),
    "rigla":     (parser_Rigla,     drugs),
}

results = run_in_chunks(tasks, drugs, chunk_size=200, max_workers=3)
data_Gorzdrav  = results.get("gorzdrav")
data_Rigla     = results.get("rigla")

normalized_data_Rigla = normalize_pharmacy_data(data_Rigla,"rigla")
normalized_data_Gorzdrav = normalize_pharmacy_data(data_Gorzdrav,"gorzdrav")

matched_Rigla = build_pharmacy_summary(normalized_data_Rigla, "rigla", drugs)
matched_Gorzdrav = build_pharmacy_summary(normalized_data_Gorzdrav, "gorzdrav", drugs)

# ==========================================
# Запись в базу данных
# ==========================================
DB_URL = os.environ.get("DB_URL")
if DB_URL:
    print("\nПодключение к базе данных...")
    engine = create_engine(DB_URL)
    today = date.today()
    
    def save_to_db(df, table_name):
        if df.empty:
            return
            
        df_export = pd.DataFrame({
            "date": today,
            "query": df["Исходное наим товара"],
            "product_name": df.iloc[:, 1],
            "price": df.iloc[:, 2],
            "url": df.iloc[:, 3]
        })
        # Защита от дублей (CardinalityViolation)
        df_export = df_export.drop_duplicates(subset=["date", "query"], keep="first")
        
        upsert_query = text(f"""
            INSERT INTO {table_name} (date, query, product_name, price, url)
            SELECT date, query, product_name, price, url
            FROM temp_table
            ON CONFLICT (date, query) 
            DO UPDATE SET 
                product_name = EXCLUDED.product_name,
                price = EXCLUDED.price,
                url = EXCLUDED.url;
        """)
        
        with engine.begin() as conn:
            df_export.to_sql("temp_table", conn, if_exists="replace", index=False)
            conn.execute(upsert_query)
            conn.execute(text("DROP TABLE temp_table;"))

    save_to_db(matched_Rigla, "price_rigla")
    print("✅ Ригла записана в базу.")

    save_to_db(matched_Gorzdrav, "price_gorzdrav")
    print("✅ Горздрав записан в базу.")

else:
    print("⚠️ Переменная DB_URL не найдена.")
