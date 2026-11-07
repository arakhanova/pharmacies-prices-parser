import sys
import os
from pathlib import Path
from datetime import date
from sqlalchemy import create_engine, text

# 1. Добавляем текущую рабочую директорию в sys.path
# Именно оттуда GitHub Actions запускает скрипт
sys.path.insert(0, os.getcwd())
# И на всякий случай папку самого скрипта
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 2. Теперь импортируем всё остальное
import pandas as pd
import re
import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# 3. Импортируем локальные модули
import parser_Rigla
import parser_Gorzdrav
import extractor

# 4. Загрузка данных
try:
    # Это сработает в GitHub Actions (файл .py)
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    # Это сработает в ноутбуке
    PROJECT_DIR = Path.cwd()
sys.path.insert(0, str(PROJECT_DIR))

df_in = pd.read_excel(PROJECT_DIR / "drugs.xlsx", header=None)
drugs = df_in[0].dropna().astype(str).str.strip().tolist()
DRUGS_TEST = drugs[:100] #тестовый срез

def run_in_chunks(tasks, drug_list, chunk_size=200, max_workers=3):
    """
    tasks: dict {name: (module, drug_list)}
    Возвращает: dict {name: pd.DataFrame} — объединённые результаты по всем чанкам.
    """
    # 1. Разбиваем каждый список на чанки
    #    (все аптеки получают одинаковые чанки, чтобы прогресс был синхронный)
    all_drugs = None
    for name, (module, drug_list) in tasks.items():
        all_drugs = drug_list
        break
    chunks = [all_drugs[i:i+chunk_size] for i in range(0, len(all_drugs), chunk_size)]
    print(f"Чанков: {len(chunks)} × {chunk_size} препаратов")

    # 2. Аккумуляторы
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

        # 3. Чистим память между чанками
        import gc
        gc.collect()

        elapsed = time.time() - t0
        eta = elapsed / ci * (len(chunks) - ci)
        print(f"  прогресс: {elapsed:.0f}s, ETA {eta:.0f}s")

    # 4. Склеиваем
    results = {}
    for name, frames in acc.items():
        if frames:
            results[name] = pd.concat(frames, ignore_index=True)
            results[name] = results[name].drop_duplicates(subset=["Ссылка"]).reset_index(drop=True)
        else:
            results[name] = pd.DataFrame()
        print(f"[{name:10s}] итого: {results[name].shape}")
    return results


tasks = {
    "gorzdrav":  (parser_Gorzdrav,  DRUGS_TEST),
    "rigla":     (parser_Rigla,     DRUGS_TEST),
    # "stolichki": (parser_Stolichki, DRUGS_TEST),
}

SCHEMAS = {
    "gorzdrav": {
        "rename": {
            "Поисковый запрос": "query",
            "Наименование": "name",
            "Цена (лучшая)": "price",
            "Цена без скидки": "price_regular",
            "Производитель": "manufacturer",
            "Действующее вещество": "active_substance",
            "Страна": "country",
            "Ссылка": "url",
        },
    },
    "rigla": {
        "rename": {
            "Поисковый запрос": "query",
            "Наименование": "name",
            "Цена": "price",
            "Цена без скидки": "price_regular",
            "Цена по карте": "price_card",
            "Производитель": "manufacturer",
            "В наличии": "in_stock",
            "Ссылка": "url",
        },
    },
}

UNIFIED_COLS = [
    "pharmacy", "query", "name", "price",
    "price_regular", "price_card",
    "manufacturer", "active_substance", "country", "in_stock",
    "url",
]

# ==========================================
# Нормализация (приведение к формату)
# ==========================================
def normalize_pharmacy_data(df, pharmacy_name):
    """
    Приводит DataFrame к единому формату.
    Возвращает новый DataFrame, ничего не сохраняет.
    """
    if df is None or df.empty:
        print(f"Данные для '{pharmacy_name}' пусты.")
        return pd.DataFrame(columns=UNIFIED_COLS)

    # 1. Переименовываем колонки по схеме
    rename_map = SCHEMAS.get(pharmacy_name, {}).get("rename", {})
    df = df.rename(columns=rename_map).copy()

    # 2. Добавляем недостающие колонки (пустыми)
    for col in UNIFIED_COLS:
        if col not in df.columns:
            df[col] = None

    # 3. Добавляем название аптеки
    df["pharmacy"] = pharmacy_name

    # 4. Оставляем только нужные колонки в правильном порядке
    df = df[UNIFIED_COLS]

    # 5. Приводим цены к числовому типу
    for col in ["price", "price_regular", "price_card"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ==========================================
# Сохранение - пока не используется
# ==========================================
def save_pharmacy_data(df, pharmacy_name, output_dir, format="parquet"):
    """
    Сохраняет DataFrame в файл.
    format: "parquet" или "pickle"
    """
    if df is None or df.empty:
        print(f"Нечего сохранять для '{pharmacy_name}'.")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format == "parquet":
        output_path = output_dir / f"{pharmacy_name}.parquet"
        df.to_parquet(output_path, index=False)
    elif format == "pickle":
        output_path = output_dir / f"{pharmacy_name}.pkl"
        df.to_pickle(output_path)
    else:
        raise ValueError(f"Неизвестный формат: {format}")

    print(f"✅ {pharmacy_name}: {len(df)} строк сохранено в {output_path.name}")
    return output_path


def build_pharmacy_summary(df, pharmacy_name, drugs_list):
    # 1. Фильтруем данные по аптеке
    pharmacy_raw = df[
        (df["pharmacy"] == pharmacy_name) & 
        (df["query"].isin(drugs_list))
    ].copy()
    
    if pharmacy_raw.empty:
        print(f"Для аптеки '{pharmacy_name}' не найдено данных")
        return pd.DataFrame({
            "Исходное наим товара": drugs_list,
            f"Товар {pharmacy_name.capitalize()}": None,
            f"Цена {pharmacy_name.capitalize()}": None,
            f"Ссылка {pharmacy_name.capitalize()}": None
        })

    # 2. Извлекаем match_key из выдачи парсера
    print(f"[{pharmacy_name.capitalize()}] Извлекаем match_key из выдачи...")
    pharmacy_raw["parsed_match_key"] = pharmacy_raw["name"].apply(
        lambda x: extractor.build_match_key(extractor.normalize_entity(extractor.extract_regex(x))) 
        if extractor.extract_regex(x) else None
    )

    # 3. Извлекаем match_key из исходного списка
    drugs_df = pd.DataFrame({"Исходное наим товара": drugs_list})
    
    def get_target_key(name):
        e = extractor.extract_regex(name)
        if e:
            return extractor.build_match_key(extractor.normalize_entity(e))
        return None
        
    drugs_df["target_match_key"] = drugs_df["Исходное наим товара"].apply(get_target_key)

    # 4. Сначала пробуем СТРОГОЕ совпадение
    print(f"[{pharmacy_name.capitalize()}] Ищем строгие совпадения...")
    matched = pd.merge(
        drugs_df,
        pharmacy_raw,
        left_on=["Исходное наим товара", "target_match_key"],
        right_on=["query", "parsed_match_key"],
        how="left"
    )

    # =========================================================================
    # СПАСЕНИЕ ДЛЯ СЛОЖНЫХ СЛУЧАЕВ (Азитромицин-Вертекс, Азопт и т.д.)
    # =========================================================================
    missing_mask = matched["name"].isna()
    
    if missing_mask.any():
        print(f"[{pharmacy_name.capitalize()}] Спасаем {missing_mask.sum()} записей по бренду и производителю...")
        
        def rescue_by_manufacturer(row):
            query_name = str(row["Исходное наим товара"]).lower()
            
            # Берем первое слово из запроса (арпефлю)
            brand_word = query_name.split()[0]
            # Убираем всё, кроме букв (если есть дефисы или точки)
            brand_word = re.sub(r"[^а-яa-z]", "", brand_word)
            
            if not brand_word:
                return pd.Series({"name": None, "price": None, "url": None})
            
            # Ищем в аптеке все товары, где в названии есть этот бренд
            candidates = pharmacy_raw[
                pharmacy_raw["name"].str.lower().str.contains(brand_word, na=False)
            ]
            
            if candidates.empty:
                return pd.Series({"name": None, "price": None, "url": None})
            
            # Берем самый дешевый из найденных
            best = candidates.sort_values("price").iloc[0]
            return pd.Series({
                "name": best["name"],
                "price": best["price"],
                "url": best["url"]
            })

        # Применяем спасение
        rescued = matched[missing_mask].apply(rescue_by_manufacturer, axis=1)
        
        # Записываем спасенные данные обратно
        matched.loc[missing_mask, "name"] = rescued["name"].values
        matched.loc[missing_mask, "price"] = rescued["price"].values
        matched.loc[missing_mask, "url"] = rescued["url"].values
    # =========================================================================

    # 5. Агрегация (берем минимальную цену)
    matched_data = (
        matched
        .sort_values("price")
        .groupby("Исходное наим товара")
        .agg({
            "name": "first",
            "price": "first",
            "url": "first"
        })
        .reset_index()
    )

    # 6. Привязка к исходному порядку
    matched_data = pd.merge(
        pd.DataFrame({"Исходное наим товара": drugs_list}),
        matched_data,
        on="Исходное наим товара",
        how="left"
    )

    # 7. Красивые названия колонок
    cap_name = pharmacy_name.capitalize()
    matched_data.columns = [
        "Исходное наим товара", 
        f"Товар {cap_name}", 
        f"Цена {cap_name}", 
        f"Ссылка {cap_name}"
    ]

    return matched_data
    
results = run_in_chunks(tasks, drugs, chunk_size=200, max_workers=3)
data_Gorzdrav  = results.get("gorzdrav")
data_Rigla     = results.get("rigla")
#data_Stolichki = results.get("stolichki")

normalized_data_Rigla = normalize_pharmacy_data(data_Rigla,"rigla")
normalized_data_Gorzdrav = normalize_pharmacy_data(data_Gorzdrav,"gorzdrav")

matched_Rigla = build_pharmacy_summary(normalized_data_Rigla, "rigla", drugs)
matched_Gorzdrav = build_pharmacy_summary(normalized_data_Gorzdrav, "gorzdrav", drugs)

DB_URL = os.environ.get("DB_URL")
if DB_URL:
    print("Подключение к базе данных...")
    engine = create_engine(DB_URL)
    today = date.today()
    
    def save_to_db(df, table_name):
        if df.empty:
            return
            
        # Переименовываем колонки DataFrame под новые имена в базе
        df_export = pd.DataFrame({
            "date": today,
            "query": df["Исходное наим товара"],
            "product_name": df.iloc[:, 1],  # Вторая колонка (Товар Ригла/Горздрав)
            "price": df.iloc[:, 2],         # Третья колонка (Цена)
            "url": df.iloc[:, 3]            # Четвертая колонка (Ссылка)
        })
        # УБИРАЕМ ДУБЛИКАТЫ ПЕРЕД ВСТАВКОЙ
        df_export = df_export.drop_duplicates(subset=["date", "query"], keep="first")
        
        # Теперь UPSERT становится элементарным
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

    # 1. Сохраняем Риглу
    save_to_db(matched_Rigla, "price_rigla")
    print("Ригла записана в базу.")

    # 2. Сохраняем Горздрав
    save_to_db(matched_Gorzdrav, "price_gorzdrav")
    print("Горздрав записан в базу.")

else:
    print("Переменная DB_URL не найдена.")
