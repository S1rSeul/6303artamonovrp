import logging
import time
from typing import Iterator

import numpy as np

import pandas as pd


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