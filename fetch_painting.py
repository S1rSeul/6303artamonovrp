import csv
import json
import os
import random


import requests


def get_painting_id(csv_path: str) -> str:
    paintings = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if (row.get('Classification') == 'Paintings'
                    and row.get('Is Public Domain') == 'True'):
                paintings.append(row.get("Object ID"))

    return random.choice(paintings)


def fetch_object_metadata(object_id: str) -> dict:
    url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def download_image(image_url: str, save_path: str) -> None:
    response = requests.get(image_url)
    response.raise_for_status()

    with open(save_path, 'wb') as f:
        f.write(response.content)


def save_metadata(data: dict, save_path: str) -> None:
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    csv_path = 'MetObjects.csv'
    output_dir = 'paintings'
    os.makedirs(output_dir, exist_ok=True)
    object_id = get_painting_id(csv_path)

    metadata = fetch_object_metadata(object_id)
    primary_image = metadata.get('primaryImage')

    img_path = os.path.join(output_dir, 'image.jpg')
    json_path = os.path.join(output_dir, 'image.json')

    download_image(primary_image, img_path)
    save_metadata(metadata, json_path)


if __name__ == '__main__':
    main()
