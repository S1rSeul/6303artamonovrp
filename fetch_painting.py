import csv
import json
import os
import random
import urllib.request


def get_painting(csv_path: str) -> dict:
    paintings = []
    with (open(csv_path, mode='r', encoding='utf-8') as f):
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get('Classification') == 'Paintings'
                    and row.get('Is Public Domain') == 'True'):
                paintings.append(row)

    return random.choice(paintings)


def fetch_object_metadata(object_id: int) -> dict:
    url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode('utf-8'))

    return data


def download_image(image_url: str, save_path: str) -> None:
    urllib.request.urlretrieve(image_url, save_path)


def save_metadata(data: dict, save_path: str) -> None:
    with open(save_path, 'w', encoding='utf-8') as path:
        json.dump(data, path, indent=2, ensure_ascii=False)


def main() -> None:
    csv_path = 'MetObjects.csv'
    output_dir = 'paintings'
    os.makedirs(output_dir, exist_ok=True)
    painting = get_painting(csv_path)

    object_id = painting.get("Object ID")

    metadata = fetch_object_metadata(object_id)
    primary_image = metadata.get('primaryImage')

    img_path = os.path.join(output_dir, 'image.jpg')
    json_path = os.path.join(output_dir, 'image.json')

    download_image(primary_image, img_path)
    save_metadata(metadata, json_path)


if __name__ == '__main__':
    main()
