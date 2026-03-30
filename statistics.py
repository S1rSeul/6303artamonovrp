import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterator, Tuple, Dict, List, Any


def compute_age(row: pd.Series) -> float:
    accession_year = float(row['accessionYear'])
    obj_begin_date = float(row['objectBeginDate'])
    if pd.isna(accession_year) or pd.isna(obj_begin_date):
        return np.nan

    age = accession_year - obj_begin_date
    if age < 0:
        return np.nan

    return age


def read_chunks(filepath: str, chunksize: int = 50_000) -> Iterator[pd.DataFrame]:
    reader = pd.read_csv(filepath, chunksize=chunksize)
    for chunk in reader:
        yield chunk


def process_chunk(chunk_iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    for chunk in chunk_iter:
        df = chunk.copy()
        df['age'] = df.apply(compute_age, axis=1)
        df.dropna(subset=['culture', 'age'], inplace=True)
        if not df.empty:
            yield df


def aggregate(processed_iter: Iterator[pd.DataFrame]) -> tuple:
    stats_aggregate = {}
    year_stats = {}

    for df in processed_iter:
        for culture, group in df.groupby('culture'):
            if culture not in stats_aggregate:
                stats_aggregate[culture] = {
                    'count': 0,
                    'sum_age': 0.0,
                    'sum_age_square': 0.0,
                    'min_accession': np.inf,
                    'max_accession': -np.inf,
                }

            cur = stats_aggregate[culture]
            cur['count'] += len(group)
            cur['sum_age'] += group['age'].sum()
            cur['sum_age_sq'] += (group['age'] ** 2).sum()
            cur['min_accession'] = min(cur['min_accession'], group['accessionYear'].min())
            cur['max_accession'] = max(cur['max_accession'], group['accessionYear'].max())

            if culture not in year_stats:
                year_stats[culture] = {}

            for year, sub in group.groupby('accessionYear'):
                if year not in year_stats[culture]:
                    year_stats[culture][year] = (0.0, 0)
                s, c = year_stats[culture][year]
                year_stats[culture][year] = (s + sub['age'].sum(), c + len(sub))

    return stats_aggregate, year_stats


def compute_statistics(stats_agg: dict) -> dict:
    stats = {}
    for culture, data in stats_agg.items():
        n = data['count']
        if n == 0:
            continue

        sum_age = data['sum_age']
        sum_age_square = data['sum_age_square']
        mean = sum_age / n
        var = (sum_age_square / n) - mean ** 2
        std = np.sqrt(var)
        se = std / np.sqrt(n)
        ci_low = mean - 1.96 * se
        ci_high = mean + 1.96 * se
        scatter_low = mean - 1.96 * std
        scatter_high = mean + 1.96 * std
        stats[culture] = {
            'count': n,
            'mean': mean,
            'std': std,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'scatter_low': scatter_low,
            'scatter_high': scatter_high,
            'min_accession': data['min_accession'],
            'max_accession': data['max_accession'],
            'range_accession': data['max_accession'] - data['min_accession'],
        }
    return stats



