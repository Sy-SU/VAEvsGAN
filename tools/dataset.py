#!/usr/bin/env python3
"""Download MNIST and arrange the extracted files into train/test directories."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import shutil
import struct
import sys
import urllib.error
import urllib.request
import zlib
from pathlib import Path


BASE_URLS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist",
    "https://ossci-datasets.s3.amazonaws.com/mnist",
)

FILES = {
    # archive name: (MD5, split, extracted file name)
    "train-images-idx3-ubyte.gz": (
        "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        "train",
        "images-idx3-ubyte",
    ),
    "train-labels-idx1-ubyte.gz": (
        "d53e105ee54ea40749a09fcbcd1e9432",
        "train",
        "labels-idx1-ubyte",
    ),
    "t10k-images-idx3-ubyte.gz": (
        "9fb629c4189551a2d022fa330f9573f3",
        "test",
        "images-idx3-ubyte",
    ),
    "t10k-labels-idx1-ubyte.gz": (
        "ec29112dd5afa0611ce80d1b7f02629c",
        "test",
        "labels-idx1-ubyte",
    ),
}

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "MNIST"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SPLIT_SIZES = {"train": 60_000, "test": 10_000}


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - MD5 is used only for file integrity.
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(filename: str, destination: Path, expected_md5: str) -> None:
    if destination.exists() and md5sum(destination) == expected_md5:
        print(f"Using existing archive: {destination}")
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    errors: list[str] = []

    for base_url in BASE_URLS:
        url = f"{base_url}/{filename}"
        try:
            print(f"Downloading {url}")
            request = urllib.request.Request(
                url, headers={"User-Agent": "mnist-dataset-downloader/1.0"}
            )
            with urllib.request.urlopen(request) as response, temporary.open("wb") as file:
                shutil.copyfileobj(response, file)

            actual_md5 = md5sum(temporary)
            if actual_md5 != expected_md5:
                raise ValueError(
                    f"checksum mismatch: expected {expected_md5}, got {actual_md5}"
                )

            temporary.replace(destination)
            return
        except (OSError, ValueError, urllib.error.URLError) as error:
            errors.append(f"{url}: {error}")
            temporary.unlink(missing_ok=True)

    raise RuntimeError(
        f"Unable to download {filename} from any mirror:\n  " + "\n  ".join(errors)
    )


def extract(archive: Path, destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        print(f"Using existing extracted file: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Extracting {archive} -> {destination}")
    try:
        with gzip.open(archive, "rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)


def write_png(path: Path, pixels: bytes, width: int, height: int) -> None:
    scanlines = b"".join(
        b"\x00" + pixels[row * width : (row + 1) * width] for row in range(height)
    )
    image = b"".join(
        (
            PNG_SIGNATURE,
            png_chunk(
                b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
            ),
            png_chunk(b"IDAT", zlib.compress(scanlines, level=9)),
            png_chunk(b"IEND", b""),
        )
    )
    path.write_bytes(image)


def exported_split_is_complete(split_dir: Path) -> bool:
    expected_count = SPLIT_SIZES.get(split_dir.name)
    labels_csv = split_dir / "labels.csv"
    images_dir = split_dir / "images"
    if expected_count is None or not labels_csv.is_file() or not images_dir.is_dir():
        return False

    with labels_csv.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
        label_count = sum(1 for _ in reader)

    image_count = sum(1 for _ in images_dir.glob("*.png"))
    return (
        header == ["image", "label"]
        and label_count == expected_count
        and image_count == expected_count
    )


def export_split(split_dir: Path, force: bool = False) -> None:
    images_idx = split_dir / "images-idx3-ubyte"
    labels_idx = split_dir / "labels-idx1-ubyte"
    labels_csv = split_dir / "labels.csv"
    legacy_marker = split_dir / ".png-export-complete"
    images_dir = split_dir / "images"

    legacy_marker.unlink(missing_ok=True)

    if exported_split_is_complete(split_dir) and not force:
        images_idx.unlink(missing_ok=True)
        labels_idx.unlink(missing_ok=True)
        print(f"Using existing PNG export: {split_dir}")
        return

    with images_idx.open("rb") as images, labels_idx.open("rb") as labels:
        image_header = images.read(16)
        label_header = labels.read(8)
        if len(image_header) != 16 or len(label_header) != 8:
            raise RuntimeError(f"Invalid IDX header in {split_dir}")

        image_magic, image_count, height, width = struct.unpack(">IIII", image_header)
        label_magic, label_count = struct.unpack(">II", label_header)
        if image_magic != 2051 or label_magic != 2049:
            raise RuntimeError(f"Invalid IDX magic number in {split_dir}")
        if image_count != label_count:
            raise RuntimeError(
                f"Image/label count mismatch in {split_dir}: "
                f"{image_count} images, {label_count} labels"
            )

        pixel_count = width * height
        expected_image_size = 16 + image_count * pixel_count
        expected_label_size = 8 + label_count
        if images_idx.stat().st_size != expected_image_size:
            raise RuntimeError(f"Invalid image IDX size: {images_idx}")
        if labels_idx.stat().st_size != expected_label_size:
            raise RuntimeError(f"Invalid label IDX size: {labels_idx}")

        images_dir.mkdir(parents=True, exist_ok=True)

        temporary_csv = labels_csv.with_suffix(".csv.part")
        digits = max(5, len(str(image_count - 1)))
        try:
            with temporary_csv.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(("image", "label"))

                for index in range(image_count):
                    pixels = images.read(pixel_count)
                    label_data = labels.read(1)
                    if len(pixels) != pixel_count or len(label_data) != 1:
                        raise RuntimeError(f"Unexpected end of IDX data in {split_dir}")

                    label = label_data[0]
                    if label > 9:
                        raise RuntimeError(
                            f"Invalid MNIST label {label} at index {index}"
                        )

                    filename = f"{index:0{digits}d}.png"
                    relative_path = Path("images") / filename
                    write_png(split_dir / relative_path, pixels, width, height)
                    writer.writerow((relative_path.as_posix(), label))

                    if (index + 1) % 10_000 == 0 or index + 1 == image_count:
                        print(
                            f"Exported {index + 1}/{image_count} PNGs to {images_dir}"
                        )

            temporary_csv.replace(labels_csv)
            for label in range(10):
                legacy_dir = images_dir / str(label)
                if legacy_dir.is_dir():
                    shutil.rmtree(legacy_dir)
        finally:
            temporary_csv.unlink(missing_ok=True)

    images_idx.unlink(missing_ok=True)
    labels_idx.unlink(missing_ok=True)


def download_mnist(data_dir: Path, force: bool = False) -> None:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    complete_splits = {
        split
        for split in SPLIT_SIZES
        if not force and exported_split_is_complete(data_dir / split)
    }

    for filename, (expected_md5, split, extracted_name) in FILES.items():
        archive = raw_dir / filename
        extracted = data_dir / split / extracted_name

        if split in complete_splits:
            extracted.unlink(missing_ok=True)
            continue

        if force:
            archive.unlink(missing_ok=True)
        download(filename, archive, expected_md5)
        extract(archive, extracted, force=force)

    for split in ("train", "test"):
        export_split(data_dir / split, force=force)

    print(f"MNIST is ready in {data_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MNIST and export train/test samples as PNG files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"destination directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="download and extract all files again",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        download_mnist(args.data_dir, force=args.force)
    except (OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
