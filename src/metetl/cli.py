import argparse
import asyncio
import json
import logging
import os
import sys

from metetl.images.processing import ImageProcessor
from metetl.analysis.aggregations import run_pipeline as run_analysis
from metetl.logging_config import configure_logging


def cmd_prepare(args):
    logging.info("Запуск подготовки метаданных...")
    processor = ImageProcessor(args.csv)
    ids = processor._load_painting_ids(args.csv, count=None)
    if not ids:
        logging.error("Не найдено ни одной подходящей картины.")
        sys.exit(1)

    import pandas as pd
    df = pd.read_csv(args.csv, dtype=str, low_memory=False)
    df = df[df['Object ID'].isin(ids)]
    cols = [
        'Object ID', 'Title', 'Artist Display Name', 'Artist Begin Date',
        'Artist End Date', 'Object Begin Date', 'Object End Date',
        'Medium', 'Dimensions', 'Classification', 'Is Public Domain'
    ]
    metadata_list = df[cols].to_dict(orient='records')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
    logging.info(f"Метаданные сохранены в {args.output} (всего {len(metadata_list)} записей)")


def cmd_process(args):
    if not os.path.exists(args.input):
        logging.error(f"Файл {args.input} не найден.")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        all_metadata = json.load(f)

    if args.num > len(all_metadata):
        logging.error(f"В JSON только {len(all_metadata)} записей, запрошено {args.num}.")
        sys.exit(1)

    import random
    random.seed(1)
    selected = random.sample(all_metadata, args.num)
    painting_ids = [item['Object ID'] for item in selected]

    processor = ImageProcessor(
        csv_path=args.csv,
        output_dir=args.output,
        max_concurrent_downloads=args.max_concurrent_downloads,
    )
    asyncio.run(processor.run_pipeline(painting_ids))

def cmd_analyze(args):
    run_analysis(
        csv_path=args.csv,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
    )

def main():
    parser = argparse.ArgumentParser(
        prog='metetl',
        description='Утилита для скачивания, обработки и анализа изображений из коллекции MET.'
    )
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')

    prepare_parser = subparsers.add_parser('prepare', help='Подготовить JSON с метаданными')
    prepare_parser.add_argument('--csv', default='MetObjects.csv', help='Путь к CSV-файлу MET')
    prepare_parser.add_argument('--output', default='to_download.json', help='Путь для сохранения JSON')
    prepare_parser.set_defaults(func=cmd_prepare)

    process_parser = subparsers.add_parser('process', help='Скачать и обработать изображения')
    process_parser.add_argument('--input', required=True, help='JSON с метаданными')
    process_parser.add_argument('--output', default='paintings', help='Папка для изображений')
    process_parser.add_argument('--num', type=int, required=True, help='Количество изображений')
    process_parser.add_argument('--max-concurrent-downloads', type=int, default=10, help='Одновременных загрузок')
    process_parser.add_argument('--csv', default='MetObjects.csv', help='Путь к CSV (не обязателен при наличии JSON)')
    process_parser.set_defaults(func=cmd_process)

    analyze_parser = subparsers.add_parser('analyze', help='Анализировать CSV и построить графики')
    analyze_parser.add_argument('--csv', required=True, help='Путь к CSV-файлу')
    analyze_parser.add_argument('--output-dir', default='plots', help='Папка для сохранения графиков')
    analyze_parser.add_argument('--chunksize', type=int, default=50000, help='Размер чанка при чтении')
    analyze_parser.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    configure_logging()
    args.func(args)


if __name__ == '__main__':
    main()
