import logging
import os
import time
from typing import Iterator, Optional

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from logging_config import configure_logging



def extract_year(series: pd.Series) -> pd.Series:
    years = pd.to_numeric(series, errors='coerce')
    mask = years.isna()

    if mask.any():
        dates = pd.to_datetime(series[mask], errors='coerce')
        years.loc[mask] = dates.dt.year

    return years.astype('Int32')


def read_chunks(filepath: str, chunksize: int = 50_000) -> Iterator[pd.DataFrame]:
    dtype = {
        'AccessionYear': 'string',
        'Object Begin Date': 'string',
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
        chunk.drop('Object Begin Date', axis=1, inplace=True)
        chunk.loc[chunk['age'] < 0, 'age'] = np.nan

        original_len = len(chunk)
        total_rows += original_len

        chunk.dropna(subset=['age'], inplace=True)
        chunk['age_square'] = chunk['age'] ** 2
        kept = len(chunk)

        elapsed = time.perf_counter() - start
        total_elapsed += elapsed
        logging.debug(f"Чанк {chunk_count}: обработано {original_len} строк, оставлено {kept}, время {elapsed:.3f} сек")

        if not chunk.empty:
            total_rows_processed += kept
            yield chunk

    logging.debug(f"Всего прочитано строк: {total_rows}, обработано строк: {total_rows_processed}, общее время обработки: {total_elapsed:.3f} сек")


def aggregate(processed_iter: Iterator[pd.DataFrame]) -> pd.DataFrame:
    logging.debug("Начало агрегации данных...")
    total_elapsed = 0.0

    stats_df = pd.DataFrame()
    chunk_counter = 0

    for df in processed_iter:
        chunk_counter += 1
        chunk_start = time.perf_counter()

        df['decade'] = (df['AccessionYear'] // 10) * 10
        chunk_stats = df.groupby('decade').agg(
            count=('age', 'count'),
            sum_age=('age', 'sum'),
            sum_age_square=('age_square', 'sum'),
        )

        if stats_df.empty:
            stats_df = chunk_stats
        else:
            stats_df = stats_df.add(chunk_stats)

        chunk_elapsed = time.perf_counter() - chunk_start
        total_elapsed += chunk_elapsed
        logging.debug(f"Агрегация чанка {chunk_counter} заняла {chunk_elapsed:.3f} сек")

    stats_df['count'] = stats_df['count'].astype('int32')

    logging.debug(f"Агрегация завершена за {total_elapsed:.3f} сек")
    return stats_df


def compute_statistics(stats_df: pd.DataFrame) -> pd.DataFrame:
    logging.debug("Расчет статистик...")
    start = time.perf_counter()

    n = stats_df['count']
    mean = stats_df['sum_age'] / n
    var = (stats_df['sum_age_square'] / n) - mean ** 2
    std = np.sqrt(var)
    se = std / np.sqrt(n)
    ci_err = 1.96 * se
    scatter_err = 1.96 * std

    stats = pd.DataFrame({
        'count': n,
        'mean': mean,
        'ci_err': ci_err,
        'scatter_err': scatter_err,
    }, dtype='float32')
    stats['count'] = stats['count'].astype('int32')

    elapsed = time.perf_counter() - start
    logging.debug(f"Расчет статистик завершен за {elapsed:.3f} сек, обработано {len(stats_df)} десятилетий")
    return stats


def plot_decade_stats(stats_df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
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

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'decade_stats.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        logging.info(f"График сохранен: {path}")
        plt.close()
    else:
        plt.show()


def plot_decade_differences(stats_df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
    decades = stats_df.index.values

    if len(decades) < 2:
        logging.warning("Недостаточно десятилетий для построения графика различий")
        return

    means = stats_df['mean'].values
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

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'decade_differences.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        logging.info(f"График сохранен: {path}")
        plt.close()
    else:
        plt.show()


def run_pipeline(csv_path: str, chunksize: int = 50_000, output_dir: Optional[str] = None) -> None:
    total_start = time.perf_counter()
    logging.info(f"Начало обработки файла: {csv_path}")

    logging.debug("Чтение и обработка чанков...")
    chunks = read_chunks(csv_path, chunksize)
    processed = process_chunk(chunks)
    stats_df = aggregate(processed)

    stats = compute_statistics(stats_df)
    stats = stats.sort_index()

    logging.debug("Формирование и вывод результатов...")

    print("\nСтатистика по десятилетиям:")
    for row in stats.itertuples():
        print(f"{row.Index}–{row.Index + 9}: {row.count} объектов, "
              f"средний возраст = {row.mean:.1f} лет, "
              f"95% ДИ: ({row.mean - row.ci_err:.1f}, {row.mean + row.ci_err:.1f}), "
              f"95% интервал рассеяния: ({row.mean - row.scatter_err:.1f}, {row.mean + row.scatter_err:.1f})")

    total_elapsed = time.perf_counter() - total_start
    logging.info(f"Общее время выполнения: {total_elapsed:.3f} сек")

    plot_decade_stats(stats, output_dir)
    plot_decade_differences(stats, output_dir)

    logging.info(f"Обработка файла {csv_path} успешно завершена")