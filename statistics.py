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

    # Строим столбцовую диаграмму

    if not stats.empty:
        idx_longest = stats['range_accession'].idxmax()
        row_longest = stats.loc[idx_longest]
        print(f"\nКультура с самой длительной историей (максимальный размах годов поступления): {idx_longest}")
        print(f"Размах: {row_longest['range_accession']:.0f} лет "
              f"(от {row_longest['min_accession']:.0f} до {row_longest['max_accession']:.0f})\n")

    # Строим временной график

    output_elapsed = time.perf_counter() - output_start
    total_elapsed = time.perf_counter() - total_start

    logging.info(f"Вывод результатов занял {output_elapsed:.3f} с")
    logging.info(f"Общее время выполнения: {total_elapsed:.3f} с")


if __name__ == "__main__":
    csv = "MetObjects.csv"
    main(csv)
