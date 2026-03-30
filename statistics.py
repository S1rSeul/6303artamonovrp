import logging
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterator


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_chunks(filepath: str, chunksize: int = 50_000) -> Iterator[pd.DataFrame]:
    reader = pd.read_csv(filepath, chunksize=chunksize, low_memory=False)
    for chunk in reader:
        yield chunk


def process_chunk(chunk_iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    chunk_count = 0
    total_rows_processed = 0
    total_elapsed = 0.0

    for chunk in chunk_iter:
        chunk_count += 1
        start = time.perf_counter()

        chunk['AccessionYear'] = pd.to_numeric(chunk['AccessionYear'], errors='coerce')
        chunk['Object Begin Date'] = pd.to_numeric(chunk['Object Begin Date'], errors='coerce')
        chunk['age'] = chunk['AccessionYear'] - chunk['Object Begin Date']
        chunk.loc[chunk['age'] < 0, 'age'] = np.nan
        original_len = len(chunk)
        chunk.dropna(subset=['Culture', 'age'], inplace=True)
        kept = len(chunk)

        elapsed = time.perf_counter() - start
        total_elapsed += elapsed
        logging.info(f"Чанк {chunk_count}: обработано {original_len} строк, оставлено {kept}, время {elapsed:.3f} сек")

        if not chunk.empty:
            total_rows_processed += kept
            yield chunk

    logging.info(f"Всего обработано строк: {total_rows_processed}, общее время обработки: {total_elapsed:.3f} сек")


def aggregate(processed_iter: Iterator[pd.DataFrame]) -> tuple:
    logging.info("Начало агрегации данных...")
    total_elapsed = 0.0

    stats_frames = []
    year_frames = []

    chunk_counter = 0
    for df in processed_iter:
        chunk_counter += 1
        chunk_start = time.perf_counter()

        culture_aggregate = df.groupby('Culture').agg(
            count=('age', 'count'),
            sum_age=('age', 'sum'),
            sum_age_square=('age', lambda x: (x ** 2).sum()),
            min_accession=('AccessionYear', 'min'),
            max_accession=('AccessionYear', 'max'),
        ).reset_index()
        stats_frames.append(culture_aggregate)

        year_aggregate = df.groupby(['Culture', 'AccessionYear'])['age'].agg(['sum', 'count']).reset_index()
        year_aggregate.columns = ['Culture', 'AccessionYear', 'sum_age', 'count']
        year_frames.append(year_aggregate)

        chunk_elapsed = time.perf_counter() - chunk_start
        total_elapsed += chunk_elapsed
        logging.info(f"Агрегация чанка {chunk_counter} заняла {chunk_elapsed:.3f} сек")

    logging.info("Объединение результатов агрегации...")
    start_merge = time.perf_counter()

    if stats_frames:
        stats_all = pd.concat(stats_frames, ignore_index=True)
        stats_df = stats_all.groupby('Culture').agg(
            count=('count', 'sum'),
            sum_age=('sum_age', 'sum'),
            sum_age_square=('sum_age_square', 'sum'),
            min_accession=('min_accession', 'min'),
            max_accession=('max_accession', 'max'),
        )
    else:
        stats_df = pd.DataFrame(columns=['count', 'sum_age', 'sum_age_square', 'min_accession', 'max_accession'])

    if year_frames:
        year_all = pd.concat(year_frames, ignore_index=True)
        year_df = year_all.groupby(['Culture', 'AccessionYear']).agg(
            sum_age=('sum_age', 'sum'),
            count=('count', 'sum'),
        ).reset_index()
    else:
        year_df = pd.DataFrame(columns=['Culture', 'AccessionYear', 'sum_age', 'count'])

    merge_elapsed = time.perf_counter() - start_merge
    total_elapsed += merge_elapsed

    logging.info(f"Агрегация завершена за {total_elapsed:.3f} сек (объединение: {merge_elapsed:.3f} сек)")
    return stats_df, year_df


def compute_statistics(stats_df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Расчет статистик...")
    start = time.perf_counter()

    stats = stats_df.copy()

    stats['mean'] = stats['sum_age'] / stats['count']
    stats['var'] = (stats['sum_age_square'] / stats['count']) - stats['mean'] ** 2
    stats['std'] = np.sqrt(stats['var'])
    stats['se'] = stats['std'] / np.sqrt(stats['count'])
    stats['ci_low'] = stats['mean'] - 1.96 * stats['se']
    stats['ci_high'] = stats['mean'] + 1.96 * stats['se']
    stats['scatter_low'] = stats['mean'] - 1.96 * stats['std']
    stats['scatter_high'] = stats['mean'] + 1.96 * stats['std']
    stats['range_accession'] = stats['max_accession'] - stats['min_accession']

    stats = stats.drop(columns=['sum_age', 'sum_age_square', 'var', 'se'])

    elapsed = time.perf_counter() - start
    logging.info(f"Расчет статистик завершен за {elapsed:.3f} сек, обработано {len(stats)} культур")
    return stats


def plot_top_n(stats_df: pd.DataFrame, top_n: int = 10):
    top = stats_df.nlargest(top_n, 'count')

    cultures = top.index.tolist()
    means = top['mean'].values
    ci_lows = top['ci_low'].values
    ci_highs = top['ci_high'].values
    scatter_lows = top['scatter_low'].values
    scatter_highs = top['scatter_high'].values

    ci_err_low = [mean - ci_low for mean, ci_low in zip(means, ci_lows)]
    ci_err_high = [ci_high - mean for mean, ci_high in zip(means, ci_highs)]
    scatter_err_low = [mean - scatter_low for mean, scatter_low in zip(means, scatter_lows)]
    scatter_err_high = [scatter_high - mean for mean, scatter_high in zip(means, scatter_highs)]

    x = np.arange(len(cultures))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, means, width=0.6, alpha=0.7, color='steelblue', label='Средний возраст')
    ax.errorbar(x, means, yerr=[ci_err_low, ci_err_high], fmt='none',
                ecolor='red', capsize=5, capthick=2, label='95% доверительный интервал')
    ax.errorbar(x, means, yerr=[scatter_err_low, scatter_err_high], fmt='none',
                ecolor='gray', capsize=5, capthick=1, alpha=0.6, label='95% интервал рассеяния')
    ax.set_xticks(x)
    ax.set_xticklabels(cultures, rotation=45, ha='right')
    ax.set_ylabel('Средний возраст при поступлении (лет)')
    ax.set_title(f'Топ-{top_n} культур по частоте встречаемости\n'
                 'Средний возраст с 95% доверительным интервалом и интервалом рассеяния')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_time_series(stats: pd.DataFrame, year_df: pd.DataFrame, rolling_window: int = 10):
    culture_longest = stats['range_accession'].idxmax()
    row = stats.loc[culture_longest]

    data = year_df[year_df['Culture'] == culture_longest].copy()

    data = data.sort_values('AccessionYear')
    data['mean_age'] = data['sum_age'] / data['count']
    data['rolling_mean'] = data['mean_age'].rolling(window=rolling_window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['AccessionYear'], data['mean_age'], 'o', markersize=3, alpha=0.5, label='Средний возраст по году')
    ax.plot(data['AccessionYear'], data['rolling_mean'], color='red', linewidth=2,
            label=f'Скользящее среднее (окно = {rolling_window})')
    ax.set_xlabel('Год поступления')
    ax.set_ylabel('Средний возраст (лет)')
    ax.set_title(f'Динамика среднего возраста объектов\nКультура: {culture_longest}\n'
                 f'Период: {row['min_accession']:.0f}–{row['max_accession']:.0f} (размах = {row['range_accession']:.0f} лет)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def main(csv_path: str, chunksize: int = 50_000, top_n: int = 10, rolling_windows: int = 10):
    total_start = time.perf_counter()
    logging.info(f"Начало обработки файла: {csv_path}")

    logging.info("Чтение и обработка чанков...")
    chunks = read_chunks(csv_path, chunksize)
    processed = process_chunk(chunks)
    stats_df, year_df = aggregate(processed)

    stats = compute_statistics(stats_df)

    output_start = time.perf_counter()
    logging.info("Формирование и вывод результатов...")

    stats_sorted = stats.sort_values('count', ascending=False)
    top = stats_sorted.head(top_n)
    print(f"\nТоп-{top_n} культур по частоте встречаемости:")
    for idx, row in top.iterrows():
        print(f"{idx}: {row['count']:.0f} объектов, "
              f"средний возраст = {row['mean']:.1f} лет, "
              f"95% доверительный интервал: ({row['ci_low']:.3f}, {row['ci_high']:.3f}), "
              f"95% интервал рассеяния: ({row['scatter_low']:.3f}, {row['scatter_high']:.3f})")

    idx_longest = stats['range_accession'].idxmax()
    row_longest = stats.loc[idx_longest]
    print(f"\nКультура с самой длительной историей (максимальный размах годов поступления): {idx_longest}")
    print(f"Размах: {row_longest['range_accession']:.0f} лет "
          f"(от {row_longest['min_accession']:.0f} до {row_longest['max_accession']:.0f})\n")

    output_elapsed = time.perf_counter() - output_start
    total_elapsed = time.perf_counter() - total_start

    logging.info(f"Вывод результатов занял {output_elapsed:.3f} с")
    logging.info(f"Общее время выполнения: {total_elapsed:.3f} с")

    plot_top_n(stats, top_n)
    plot_time_series(stats, year_df, rolling_windows)

if __name__ == "__main__":
    csv = "MetObjects.csv"
    main(csv)
