import sys
import os
from pathlib import Path
from datetime import date
from sqlalchemy import create_engine, text
import pandas as pd
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. Настройка путей
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
drugs = df_in[0].dropna().astype(str).str.strip().tolist()[250:351]

# ==========================================
# Парсинг
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
                    df = future_result = fut.result()
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
# СТРОГИЙ МАТЧИНГ (ровно по вашей логике)
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
            "name": None,
            "price": None,
            "url": None
        })

    def get_entity_data(x):
        e = extractor.extract_regex(x)
        if not e:
            return {"brand": None, "form": None, "dose": None, "pack": None, "volume": None}
        norm = extractor.normalize_entity(e)
        return {
            "brand": norm.get("brand_norm"),
            "form": norm.get("form"),
            "dose": norm.get("dose_norm"),
            "pack": norm.get("pack_count"),
            "volume": norm.get("volume_ml")
        }

    # 1. Вытягиваем данные из парсинга
    pharm_ents = pharmacy_raw["name"].apply(get_entity_data)
    pharmacy_raw["p_brand"] = pharm_ents.apply(lambda x: x["brand"])
    pharmacy_raw["p_form"] = pharm_ents.apply(lambda x: x["form"])
    pharmacy_raw["p_dose"] = pharm_ents.apply(lambda x: x["dose"])
    pharmacy_raw["p_pack"] = pharm_ents.apply(lambda x: x["pack"])
    pharmacy_raw["p_volume"] = pharm_ents.apply(lambda x: x["volume"])

    # 2. Вытягиваем данные из исходного списка
    drugs_df = pd.DataFrame({"Исходное наим товара": drugs_list})
    target_ents = drugs_df["Исходное наим товара"].apply(get_entity_data)
    drugs_df["t_brand"] = target_ents.apply(lambda x: x["brand"])
    drugs_df["t_form"] = target_ents.apply(lambda x: x["form"])
    drugs_df["t_dose"] = target_ents.apply(lambda x: x["dose"])
    drugs_df["t_pack"] = target_ents.apply(lambda x: x["pack"])
    drugs_df["t_volume"] = target_ents.apply(lambda x: x["volume"])

    # 3. Сливаем по бренду
    merged = pd.merge(
        drugs_df, 
        pharmacy_raw, 
        left_on="t_brand", 
        right_on="p_brand", 
        how="inner"
    )

    # 4. Строгая фильтрация по всем полям
    form_mask = (merged["t_form"].isna()) | (merged["t_form"] == merged["p_form"])
    dose_mask = (merged["t_dose"].isna()) | (merged["t_dose"] == merged["p_dose"])
    pack_mask = (merged["t_pack"].isna()) | (merged["t_pack"] == merged["p_pack"])
    vol_mask = (merged["t_volume"].isna()) | (merged["t_volume"] == merged["p_volume"])

    strict_matches = merged[form_mask & dose_mask & pack_mask & vol_mask].copy()

    # 5. Выбираем самый дешевый
    if not strict_matches.empty:
        best_offers = (
            strict_matches
            .sort_values("price")
            .groupby("Исходное наим товара")
            .agg({
                "name": "first",
                "price": "first",
                "url": "first"
            })
            .reset_index()
        )
    else:
        best_offers = pd.DataFrame(columns=["Исходное наим товара", "name", "price", "url"])

    # 6. Привязываем к исходному списку
    final_table = pd.merge(
        pd.DataFrame({"Исходное наим товара": drugs_list}),
        best_offers,
        on="Исходное наим товара",
        how="left"
    )

    found = final_table["name"].notna().sum()
    print(f"✅ [{pharmacy_name.capitalize()}] Найдено: {found} из {len(drugs_list)} ({found/len(drugs_list):.1%})")

    return final_table

# ==========================================
# ЗАПУСК
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
# ЗАПИСЬ В БД
# ==========================================
DB_URL = os.environ.get("DB_URL")
if DB_URL:
    print("\nПодключение к базе данных...")
    engine = create_engine(DB_URL)
    today = date.today()
    
    def save_to_db(df, table_name):
        if df.empty:
            return
            
        # --- ЗАЩИТА ---
        ALLOWED_TABLES = {"price_rigla", "price_gorzdrav"}
        if table_name not in ALLOWED_TABLES:
            raise ValueError(f"Недопустимая таблица: {table_name}")
        
        df_export = pd.DataFrame({
            "date": today,
            "query": df["Исходное наим товара"],
            "product_name": df["name"],
            "price": df["price"],
            "url": df["url"]
        })
        # Защита от дублей
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
