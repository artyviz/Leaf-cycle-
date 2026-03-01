import os
import time
import urllib.request
import requests
from PIL import Image

SPECIES = {
    'oak':   'Quercus',
    'maple': 'Acer',
    'birch': 'Betula',
}

LIFECYCLE_MONTHS = {
    'bud_emergence':       '3,4',
    'expansion_maturity':  '6,7',
    'senescence':          '9,10',
    'abscission':          '11,12',
}

IMAGES_PER_CLASS = 200
INAT_API = 'https://api.inaturalist.org/v1/observations'


def is_valid_image(path):
    try:
        with Image.open(path) as img:
            if img.format not in ('JPEG', 'PNG', 'BMP'):
                return False
            img.verify()
        return True
    except Exception:
        return False


def download_class(species_name, taxon_query, stage, months, save_dir, target):
    os.makedirs(save_dir, exist_ok=True)
    existing = len([f for f in os.listdir(save_dir) if f.endswith(('.jpg', '.png'))])
    if existing >= target:
        print(f"  [SKIP] {species_name}/{stage} already has {existing} images.")
        return existing

    saved = existing
    page = 1
    print(f"  Downloading {species_name}/{stage} (target={target})...")

    while saved < target:
        params = {
            'taxon_name':    taxon_query,
            'quality_grade': 'research',
            'photos':        'true',
            'month':         months,
            'per_page':      100,
            'page':          page,
            'order_by':      'votes',
        }
        try:
            resp = requests.get(INAT_API, params=params, timeout=15)
            resp.raise_for_status()
            results = resp.json().get('results', [])
        except Exception as e:
            print(f"    API error: {e}")
            break

        if not results:
            break

        for obs in results:
            if saved >= target:
                break
            try:
                raw_url = obs['photos'][0]['url']
                img_url = raw_url.replace('square', 'medium')
                fname = os.path.join(save_dir, f"{species_name}_{stage}_{saved:04d}.jpg")
                urllib.request.urlretrieve(img_url, fname)
                if is_valid_image(fname):
                    saved += 1
                else:
                    os.remove(fname)
            except Exception:
                pass

        page += 1
        time.sleep(0.5)

    print(f"    => {saved} valid images saved to {save_dir}")
    return saved


def main():
    print("=" * 55)
    print("  iNaturalist Tree Lifecycle Dataset Downloader")
    print("=" * 55)
    total = 0
    for species_name, taxon in SPECIES.items():
        print(f"\n[Species] {species_name.upper()} ({taxon})")
        for stage, months in LIFECYCLE_MONTHS.items():
            save_dir = os.path.join('data', species_name, stage)
            n = download_class(species_name, taxon, stage, months, save_dir, IMAGES_PER_CLASS)
            total += n

    print("\n" + "=" * 55)
    print(f"  Download Complete! Total valid images: {total}")
    print(f"  Expected structure: data/{{species}}/{{stage}}/")
    print("=" * 55)

    print("\nDataset Summary:")
    for species_name in SPECIES:
        species_dir = os.path.join('data', species_name)
        if os.path.isdir(species_dir):
            for stage in LIFECYCLE_MONTHS:
                stage_dir = os.path.join(species_dir, stage)
                count = len([f for f in os.listdir(stage_dir) if f.endswith(('.jpg','.png'))]) if os.path.isdir(stage_dir) else 0
                print(f"  {species_name:8s} / {stage:22s}: {count:4d} images")


if __name__ == '__main__':
    main()
