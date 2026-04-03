import logging
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterator


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_year(series: pd.Series) -> pd.Series:
    years = pd.to_numeric(series, errors='coerce')
    mask = years.isna()
    if mask.any():
        dates = pd.to_datetime(series[mask], errors='coerce')
        years.loc[mask] = dates.dt.year
    return years.astype('Int32')


def read_chunks(filepath: str, chunksize: int = 50_000) -> Iterator[pd.DataFrame]:
    dtype = {
        'AccessionYear' : 'string',
        'Object Begin Date' : 'string',
    }
    usecols = ['AccessionYear', 'Object Begin Date']
    reader = pd.read_csv(filepath, chunksize=chunksize, low_memory=False, dtype=dtype, usecols=usecols)
    for chunk in reader:
        yield chunk


def process_chunk(chunk_iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    chunk_count = 0
    total_rows = 0
    total_rows_processed = 0
    total_elapsed = 0.0

    for chunk in chunk_iter:
        chunk_count += 1
        start = time.perf_counter()

        chunk['AccessionYear'] = extract_year(chunk['AccessionYear'])
        chunk['Object Begin Date'] = extract_year(chunk['Object Begin Date'])
        chunk['age'] = chunk['AccessionYear'] - chunk['Object Begin Date']
        chunk = chunk.drop('Object Begin Date', axis=1)
        chunk.loc[chunk['age'] < 0, 'age'] = np.nan

        original_len = len(chunk)
        total_rows += original_len

        chunk.dropna(subset=['age'], inplace=True)
        chunk['age_square'] = (chunk['age'] ** 2).astype('Int32')
        kept = len(chunk)

        elapsed = time.perf_counter() - start
        total_elapsed += elapsed
        logging.info(f"Чанк {chunk_count}: обработано {original_len} строк, оставлено {kept}, время {elapsed:.3f} сек")

        if not chunk.empty:
            total_rows_processed += kept
            yield chunk

    logging.info(f"Всего прочитано строк: {total_rows}, обработано строк: {total_rows_processed}, общее время обработки: {total_elapsed:.3f} сек")

def aggregate(processed_iter: Iterator[pd.DataFrame]) -> pd.DataFrame:
    logging.info("Начало агрегации данных...")
    total_elapsed = 0.0

    stats_frames = []
    chunk_counter = 0

    for df in processed_iter:
        chunk_counter += 1
        chunk_start = time.perf_counter()

        df['decade'] = (df['AccessionYear'] // 10) * 10

        chunk_stats = df.groupby('decade', as_index=False).agg(
            count=('age', 'count'),
            sum_age=('age', 'sum'),
            sum_age_square=('age_square', 'sum'),
        )
        stats_frames.append(chunk_stats)

        chunk_elapsed = time.perf_counter() - chunk_start
        total_elapsed += chunk_elapsed
        logging.info(f"Агрегация чанка {chunk_counter} заняла {chunk_elapsed:.3f} сек")

    logging.info("Объединение результатов агрегации...")
    start_merge = time.perf_counter()

    stats_all = pd.concat(stats_frames, ignore_index=True)
    stats_df = stats_all.groupby('decade').agg(
        count=('count', 'sum'),
        sum_age=('sum_age', 'sum'),
        sum_age_square=('sum_age_square', 'sum'),
    )
    stats_df['count'] = stats_df['count'].astype('Int32')

    merge_elapsed = time.perf_counter() - start_merge
    total_elapsed += merge_elapsed

    logging.info(f"Агрегация завершена за {total_elapsed:.3f} сек (объединение: {merge_elapsed:.3f} сек)")
    return stats_df


def compute_statistics(stats_df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Расчет статистик...")
    start = time.perf_counter()

    stats_df['mean'] = stats_df['sum_age'] / stats_df['count']
    stats_df['var'] = (stats_df['sum_age_square'] / stats_df['count']) - stats_df['mean'] ** 2
    stats_df['std'] = np.sqrt(stats_df['var'])
    stats_df['se'] = stats_df['std'] / np.sqrt(stats_df['count'])
    stats_df['ci_err'] = 1.96 * stats_df['se']
    stats_df['scatter_err'] = 1.96 * stats_df['std']

    stats_df = stats_df.drop(columns=['sum_age', 'sum_age_square', 'var', 'std', 'se'])

    elapsed = time.perf_counter() - start
    logging.info(f"Расчет статистик завершен за {elapsed:.3f} сек, обработано {len(stats_df)} десятилетий")
    return stats_df


def plot_decade_stats(stats_df: pd.DataFrame):
    decades = stats_df.index.values
    means = stats_df['mean'].values
    ci_errs = stats_df['ci_err'].values
    scatter_errs = stats_df['scatter_err'].values

    x = np.arange(len(decades))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x, means, width=0.6, alpha=0.7, color='steelblue', label='Средний возраст')
    ax.errorbar(x, means, yerr=ci_errs, fmt='none',
                ecolor='red', capsize=5, capthick=2, label='95% доверительный интервал')
    ax.errorbar(x, means, yerr=scatter_errs, fmt='none',
                ecolor='gray', capsize=5, capthick=1, alpha=0.6, label='95% интервал рассеяния')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d)}–{int(d)+9}" for d in decades], rotation=45, ha='right')
    ax.set_ylabel('Средний возраст при поступлении (лет)')
    ax.set_title('Средний возраст приобретённых объектов по десятилетиям\nс 95% доверительным интервалом и интервалом рассеяния')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_decade_differences(stats_df: pd.DataFrame):
    decades = stats_df.index.values
    means = stats_df['mean'].values

    if len(decades) < 2:
        logging.warning("Недостаточно десятилетий для построения графика различий.")
        return

    diffs = np.diff(means)
    diff_decades = decades[1:]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(diff_decades, diffs, width=8, alpha=0.7, color='coral', edgecolor='black')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Десятилетие')
    ax.set_ylabel('Изменение среднего возраста (лет)')
    ax.set_title('Динамика изменения среднего возраста приобретаемых объектов\n(отличие от предыдущего десятилетия)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def main(csv_path: str, chunksize: int = 50_000):
    total_start = time.perf_counter()
    logging.info(f"Начало обработки файла: {csv_path}")

    logging.info("Чтение и обработка чанков...")
    chunks = read_chunks(csv_path, chunksize)
    processed = process_chunk(chunks)
    stats_df = aggregate(processed)

    stats = compute_statistics(stats_df)
    stats = stats.sort_index()

    logging.info("Формирование и вывод результатов...")

    print("\nСтатистика по десятилетиям:")
    for decade, row in stats.iterrows():
        print(f"{decade}–{decade + 9}: {row['count']} объектов, "
              f"средний возраст = {row['mean']:.1f} лет, "
              f"95% ДИ: ({row['mean'] - row['ci_err']:.1f}, {row['mean'] + row['ci_err']:.1f}), "
              f"95% интервал рассеяния: ({row['mean'] - row['scatter_err']:.1f}, {row['mean'] + row['scatter_err']:.1f})")

    total_elapsed = time.perf_counter() - total_start
    logging.info(f"Общее время выполнения: {total_elapsed:.3f} сек")

    plot_decade_stats(stats)
    plot_decade_differences(stats)

if __name__ == "__main__":
    csv = "MetObjects.csv"
    main(csv)
