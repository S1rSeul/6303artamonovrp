import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterator, Tuple, Dict, List, Any


def compute_age(row: pd.Series) -> float:
    accession_year = pd.to_numeric(row['AccessionYear'], errors='coerce')
    obj_begin_date = pd.to_numeric(row['Object Begin Date'], errors='coerce')
    if pd.isna(accession_year) or pd.isna(obj_begin_date):
        return np.nan

    age = accession_year - obj_begin_date
    if age < 0:
        return np.nan

    return float(age)


def read_chunks(filepath: str, chunksize: int = 50_000) -> Iterator[pd.DataFrame]:
    reader = pd.read_csv(filepath, chunksize=chunksize, low_memory=False)
    for chunk in reader:
        yield chunk


def process_chunk(chunk_iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    for chunk in chunk_iter:
        df = chunk.copy()
        df['age'] = df.apply(compute_age, axis=1)
        df['AccessionYear'] = pd.to_numeric(df['AccessionYear'], errors='coerce')
        df['Object Begin Date'] = pd.to_numeric(df['Object Begin Date'], errors='coerce')
        df.dropna(subset=['Culture', 'age'], inplace=True)
        if not df.empty:
            yield df


def aggregate(processed_iter: Iterator[pd.DataFrame]) -> tuple:
    stats_aggregate = {}
    year_stats = {}

    for df in processed_iter:
        for culture, group in df.groupby('Culture'):
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
            cur['sum_age_square'] += (group['age'] ** 2).sum()
            cur['min_accession'] = min(cur['min_accession'], group['AccessionYear'].min())
            cur['max_accession'] = max(cur['max_accession'], group['AccessionYear'].max())

            if culture not in year_stats:
                year_stats[culture] = {}

            for year, sub in group.groupby('AccessionYear'):
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


def main(csv_path: str, chunksize: int = 50_000, top_n: int = 10, rolling_windows: int = 10):
    chunks = read_chunks(csv_path, chunksize)
    processed = process_chunk(chunks)
    stats_aggregate, year_stats = aggregate(processed)

    stats = compute_statistics(stats_aggregate)

    sorted_by_count = sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True)
    top10 = sorted_by_count[:top_n]
    print("\nТоп-10 культур по частоте встречаемости:")
    for i, (cult, data) in enumerate(top10, 1):
        print(f"{i}. {cult}: {data['count']} объектов, средний возраст = {data['mean']:.1f} лет")

    # Строим столбцовую диаграмму

    culture_longest = max(stats.items(), key=lambda x: x[1]['range_accession'])[0]
    range_val = stats[culture_longest]['range_accession']
    print(f"\nКультура с самой длительной историей (максимальный размах годов поступления): {culture_longest}")
    print(
        f"Размах: {range_val:.0f} лет (от {stats[culture_longest]['min_accession']:.0f} "
        f"до {stats[culture_longest]['max_accession']:.0f})")

    # Строим временной график


if __name__ == "__main__":
    csv = "MetObjects.csv"
    main(csv)
