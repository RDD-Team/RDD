import argparse
import csv
import shutil
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile


def organize_project(folder):
    original = folder / "도로장애물·표면 인지 영상(수도권 외)"
    raw = folder / "raw"
    metadata = folder / "metadata"
    root_csv = folder / "images.csv"

    if original.exists() and raw.exists():
        raise ValueError("원본 폴더와 raw 폴더가 둘 다 있어요. 어느 쪽에 ZIP이 있는지 확인해 주세요.")
    if not original.is_dir() and not raw.is_dir():
        raise ValueError("p_project 폴더에 스크립트를 넣어 주세요. 원본 이미지 폴더를 찾지 못했어요.")
    if metadata.exists() and not metadata.is_dir():
        raise ValueError("metadata라는 이름의 파일이 있어 폴더를 만들 수 없어요.")

    metadata_csv = list(metadata.rglob("images.csv"))
    if not metadata_csv and not root_csv.is_file():
        raise ValueError("p_project와 metadata 폴더에서 images.csv를 찾지 못했어요.")

    if original.is_dir():
        original.rename(raw)
        print("원본 이미지 폴더 → raw")

    if root_csv.is_file():
        if metadata_csv:
            print("metadata에 images.csv가 이미 있어요. 바깥의 CSV는 그대로 보관했어요.")
        else:
            metadata.mkdir(exist_ok=True)
            root_csv.rename(metadata / root_csv.name)
            print("images.csv → metadata")

    print(f"폴더 정리 완료 → {folder}")


def extract_images(folder):
    output_folder = folder / "night_val_41776"

    metadata_files = sorted((folder / "metadata").rglob("images.csv"))
    if not metadata_files and (folder / "images.csv").is_file():
        metadata_files = [folder / "images.csv"]
    if len(metadata_files) != 1:
        raise ValueError("metadata 폴더 안에 images.csv가 하나 있는지 확인해 주세요.")

    archives = list(folder.rglob("1.Frontback_N02.zip"))
    if not archives:
        raise ValueError("아래 폴더에서 1.Frontback_N02.zip을 찾지 못했습니다.")

    with metadata_files[0].open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        names = {
            row["image"]
            for row in rows
            if row["split"] == "val"
            and row["time"] == "Night"
            and row["img_key"] == "41776"
        }
    if not names:
        raise ValueError("images.csv에 41776번 야간 검증 이미지가 없습니다.")
    if any(PureWindowsPath(name).name != name or not name.endswith(".png") for name in names):
        raise ValueError("images.csv에서 예상과 다른 이미지 파일명을 발견했습니다.")

    print(f"팀 검증셋에서 찾은 야간 이미지: {len(names)}장", flush=True)
    most_matches = 0
    for archive_path in sorted(archives, key=lambda path: path.stat().st_size, reverse=True):
        with ZipFile(archive_path) as archive:
            matches = {}
            for item in archive.infolist():
                name = PureWindowsPath(item.filename).name
                if item.is_dir() or name not in names:
                    continue
                if name in matches:
                    raise ValueError(f"ZIP에 같은 이미지 이름이 두 번 나옵니다: {name}")
                matches[name] = item

            most_matches = max(most_matches, len(matches))
            if len(matches) != len(names):
                continue

            with TemporaryDirectory(dir=folder) as temp_folder:
                temp_folder = Path(temp_folder)
                for name, item in matches.items():
                    with archive.open(item) as source, (temp_folder / name).open("wb") as target:
                        shutil.copyfileobj(source, target)

                output_folder.mkdir(exist_ok=True)
                for name in matches:
                    (temp_folder / name).replace(output_folder / name)

        print(f"완료: {len(matches)}장 저장 → {output_folder}")
        return

    raise ValueError(
        f"ZIP에서 {len(names) - most_matches}장을 찾지 못했습니다. "
        "파일키 41776의 10.41 GB 이미지 ZIP을 받았는지 확인해 주세요."
    )


def main():
    parser = argparse.ArgumentParser(description="RDD 프로젝트의 야간 검증 이미지 준비")
    parser.add_argument("--organize", action="store_true", help="원본 ZIP과 CSV 폴더 정리")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    if args.organize:
        organize_project(folder)
    else:
        extract_images(folder)


if __name__ == "__main__":
    try:
        main()
    except (OSError, BadZipFile, ValueError, KeyError) as error:
        raise SystemExit(f"작업 중단: {error}") from None
