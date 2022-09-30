import hashlib
import shutil
from pathlib import Path
from typing import Set

import requests
from tqdm import tqdm

ZENODO_ENTRY_POINT = "https://zenodo.org/api"
RECORDS_ENTRY_POINT = f"{ZENODO_ENTRY_POINT}/records/"

CHUNK_SIZE = 65536


class DownloadError(Exception):
    pass


def download_file(url: str, save_dir: Path, total_bytes: int) -> Path:
    """Downloads large files from the given URL.

    From: https://stackoverflow.com/a/16696317

    :param url: The URL of the file.
    :param save_dir: The directory where the file should be saved.
    :param total_bytes: The total bytes of the file.
    :return: The path to the downloaded file.
    """
    local_filename = save_dir / url.split('/')[-1]
    print(f"Starting download from {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            iters = total_bytes // CHUNK_SIZE
            for chunk in tqdm(r.iter_content(chunk_size=CHUNK_SIZE),
                              total=iters):
                f.write(chunk)

    return local_filename


def file_md5(filename: Path) -> str:
    """Computes the MD5 hash of a given file"""
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(32768), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def zenodo_download(record_id: str, filenames_to_download: Set[str],
                    save_dir: Path) -> None:
    """Downloads the given files from the given Zenodo record.

    :param record_id: The ID of the record.
    :param filenames_to_download: The files to download from the record.
    :param save_dir: The directory where the files should be saved.
    """
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    url = f"{RECORDS_ENTRY_POINT}/{record_id}"
    res = requests.get(url)
    files = res.json()["files"]
    files_to_download = list(
        filter(lambda file: file["key"] in filenames_to_download, files))

    for file in files_to_download:
        if (save_dir / file["key"]).exists():
            continue
        file_url = file["links"]["self"]
        file_checksum = file["checksum"].split(":")[-1]
        filename = download_file(file_url, save_dir, file["size"])
        if file_md5(filename) != file_checksum:
            raise DownloadError(
                "The hash of the downloaded file does not match"
                " the expected one.")
        print("Download finished, extracting...")
        shutil.unpack_archive(filename,
                              extract_dir=save_dir,
                              format=file["type"])
        print("Downloaded and extracted.")
