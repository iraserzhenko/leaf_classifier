import zipfile
from pathlib import Path

import fire

DATASET_SLUG = "abdallahalidev/plantvillage-dataset"
DEFAULT_DATA_DIR = Path("data/plantvillage")


def download_data(data_dir: str = str(DEFAULT_DATA_DIR)) -> None:
    target = Path(data_dir)
    if target.exists() and any(target.iterdir()):
        print(f"[INFO] Data directory already populated: {target}. Skipping download.")
        return

    try:
        import kaggle
    except ImportError as exc:
        raise RuntimeError(
            "kaggle package not installed. Run: pip install kaggle\n"
            "Also set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        ) from exc

    target.mkdir(parents=True, exist_ok=True)
    zip_dir = target.parent / "_downloads"
    zip_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading {DATASET_SLUG} ...")
    kaggle.api.dataset_download_files(
        DATASET_SLUG,
        path=str(zip_dir),
        unzip=False,
    )

    zip_files = list(zip_dir.glob("*.zip"))
    if not zip_files:
        raise RuntimeError("Download completed but no zip file found.")

    zip_path = zip_files[0]
    print(f"[INFO] Extracting {zip_path} -> {target} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(target.parent))

    extracted = target.parent / "PlantVillage"
    if extracted.exists() and not target.exists():
        extracted.rename(target)

    print(f"[INFO] Dataset ready at: {target}")


if __name__ == "__main__":
    fire.Fire(download_data)
