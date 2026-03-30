from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterator


def read_chunks(filepath: str, chunksize: int = 50_000) -> Iterator[pd.DataFrame]:
    reader = pd.read_csv(filepath, chunksize=chunksize, low_memory=False)
    for chunk in reader:
        yield chunk


def process_chunk(chunk_iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    for chunk in chunk_iter:
        chunk['AccessionYear'] = pd.to_numeric(chunk['AccessionYear'], errors='coerce')
        chunk['Object Begin Date'] = pd.to_numeric(chunk['Object Begin Date'], errors='coerce')
        chunk['age'] = chunk['AccessionYear'] - chunk['Object Begin Date']
        chunk.loc[chunk['age'] < 0, 'age'] = np.nan
        chunk.dropna(subset=['Culture', 'age'], inplace=True)
        if not chunk.empty:
            yield chunk


def aggregate(processed_iter: Iterator[pd.DataFrame]) -> tuple:
    stats_aggregate = {}
    year_stats = defaultdict(lambda: defaultdict(lambda: (0.0, 0)))

    for df in processed_iter:
        culture_aggregate = df.groupby('Culture').agg(
            count=('age', 'count'),
            sum_age=('age', 'sum'),
            sum_age_square=('age', lambda x: (x ** 2).sum()),
            min_accession=('AccessionYear', 'min'),
            max_accession=('AccessionYear', 'max'),
        ).to_dict(orient='index')

        for culture, vals in culture_aggregate.items():
            if culture not in stats_aggregate:
                stats_aggregate[culture] = vals
            else:
                cur = stats_aggregate[culture]
                cur['count'] += vals['count']
                cur['sum_age'] += vals['sum_age']
                cur['sum_age_square'] += vals['sum_age_square']
                cur['min_accession'] = min(cur['min_accession'], vals['min_accession'])
                cur['max_accession'] = max(cur['max_accession'], vals['max_accession'])

        year_aggregate = df.groupby(['Culture', 'AccessionYear'])['age'].agg(['sum', 'count']).reset_index()
        for _, row in year_aggregate.iterrows():
            culture, year, sum_age, cnt = row
            s, c = year_stats[culture][year]
            year_stats[culture][year] = (s + sum_age, c + cnt)

    return stats_aggregate, year_stats


def compute_statistics(stats_aggregate: dict) -> dict:
    stats = {}
    for culture, data in stats_aggregate.items():
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
